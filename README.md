## Tacotron-LPC

### 0. Requirements 
* [pylpcnet](http://120.131.1.206:8090/tts/pylpcnet)

### 1.After git clone(if uing proxyhop)
```
  proxyhop git submodule init
  proxyhop git submodule update
```
### 1. After git clone(if not use proxyhop,change the ip of respo)
```
  cd LPC-tacotron
  vi .git/config 
```
  change "url = git@120.131.1.206:tts/textnorm.git" into "url = git@172.31.0.15:8090/tts/textnorm.git"

### 2. Preprocess dataset
* `python preprocess.py`

### 3. Pretrained Model
```
    python eval.py --checkpoint pretrained/model.ckpt
```
# lpc-tracotron
