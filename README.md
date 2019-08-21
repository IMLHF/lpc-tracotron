## Tacotron-LPC

Implemented paper:
- J.-M. Valin, J. Skoglund, et al. [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Proc. International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, arXiv:1810.11846, 2019.

- Y. Wang, RJ Skerry-Ryan, D. Stanton, et al. [Tacotron: Towards end-to-end speech synthesis](https://arxiv.org/abs/1703.10135), arXiv preprint arXiv:1703.10135, 2017.

### 0. Requirements 
* [pylpcnet](https://github.com/IMLHF/pylpcnet)

### 1. Preprocess dataset
* `python preprocess.py`

### 2. Pretrained Model
```
    python eval.py --checkpoint pretrained/model.ckpt
```
