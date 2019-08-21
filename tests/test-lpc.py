import lpcnet
from util import audio
import numpy as np
import soundfile as sf
import sys

if __name__=='__main__':
    argv = sys.argv
    # test lpc utils
    features = np.load(argv[1])
    lpc_synthesizer = lpcnet.Synthesizer()
    wav = lpc_synthesizer.synthesis(features)
    wav = wav.astype(np.float32) / 32768
    sf.write('test.wav', wav, 16000)
