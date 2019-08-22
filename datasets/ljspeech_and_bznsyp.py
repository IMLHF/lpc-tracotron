from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
import lpcnet


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x, metadata_name='metadata.csv'):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
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
    with open(os.path.join(in_dir, metadata_name), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', parts[0])
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

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # trim silence
    wav = audio.trim_silence(wav)

    # convert wav to 16bit int
    wav = wav / np.max(np.abs(wav)) * 32768 * 0.9
    wav = wav.astype(np.int16)

    # extract LPC feature
    extractor = lpcnet.FeatureExtractor()
    feat = extractor.compute_feature(wav)
    n_frames = feat.shape[0]

    # write the lpc feature to disk
    feature_filename = 'ljspeech-lpc-%05d.npy' % index
    np.save(os.path.join(out_dir, feature_filename), feat, allow_pickle=False)

    # Return a tuple describing this training example:
    return (feature_filename, n_frames, text)
