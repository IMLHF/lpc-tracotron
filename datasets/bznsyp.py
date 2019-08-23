from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
import lpcnet
from util import logmmse
from hparams import hparams
from textnorm import get_pinyin


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the BZNSYP (Chinese Standard Mandarin Speech Copus | 10000 Sentences) dataset from
       a given input path into a given output directory.
       dataset: https://www.data-baker.com/open_source.html

      Args:
        in_dir: The directory where you have downloaded the BZNSYP dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    prosodylabeling_f = os.path.join("ProsodyLabeling", "000001-010000.txt")
    with open(os.path.join(in_dir, prosodylabeling_f), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 2 != 0:
                continue
            parts = line.strip().split()
            wav_path = os.path.join(in_dir, 'Wave', parts[0]+".wav")
            text = parts[1]
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, wav_path, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # text to pinyin
    text = text.replace("#1", "").replace("#2", "").replace("#3", "").replace("#4", "")
    pinyin = " ".join(get_pinyin(text))

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    wav = wav / np.max(np.abs(wav)) * 0.9 # norm

    # denoise
    if hparams.mmse_denoise_by_bothEndOfAudio and len(wav) > hparams.sample_rate*(hparams.length_as_noise*2+0.1):
      noise_wav = np.concatenate([wav[:int(hparams.sample_rate*hparams.length_as_noise)],
                                  wav[-int(hparams.sample_rate*hparams.length_as_noise):]])
      profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
      wav = logmmse.denoise(wav, profile, eta=0)

    # trim silence
    wav = audio.trim_silence(wav, hparams.trim_top_db) # top_db=30 for aishell, 60 for BZNSYP
    # audio.save_wav(wav, wav_path.replace(".wav", "_trimed.wav"))

    # convert wav to 16bit int
    wav *= 32768
    wav = wav.astype(np.int16)

    # extract LPC feature
    extractor = lpcnet.FeatureExtractor()
    feat = extractor.compute_feature(wav)
    n_frames = feat.shape[0]

    # write the lpc feature to disk
    feature_filename = 'biaobei-lpc-feat-%05d.npy' % index
    np.save(os.path.join(out_dir, feature_filename), feat, allow_pickle=False)

    # Return a tuple describing this training example:
    return (feature_filename, n_frames, pinyin)
