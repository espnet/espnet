# ESPnet: end-to-end speech processing toolkit

[![Build Status](https://travis-ci.org/espnet/espnet.svg?branch=master)](https://travis-ci.org/espnet/espnet)

ESPnet is an end-to-end speech processing toolkit, mainly focuses on end-to-end speech recognition.
ESPnet uses [chainer](https://chainer.org/) and [pytorch](http://pytorch.org/) as a main deep learning engine, 
and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for speech recognition and other speech processing experiments.


## Key Features

- Hybrid CTC/attention based end-to-end ASR 
  - Fast/accurate training with CTC/attention multitask training
  - CTC/attention joint decoding to boost monotonic alignment decoding
- Encoder: VGG-like CNN + BLSTM or pyramid BLSTM
- Attention: Dot product or location-aware attention
- Incorporate RNNLM/LSTMLM trained only with text data
- Flexible network architecture thanks to chainer and pytorch
- Kaldi style complete recipe 
  - Support numbers of ASR benchmarks (WSJ, Switchboard, CHiME-4, Librispeech, TED, CSJ, AMI, HKUST, Voxforge, etc.)
- State-of-the-art performance in Japanese/Chinese benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- Moderate performance in standard English benchmarks

## Installation

Install Kaldi, Python libraries and other required tools using system python and virtualenv
```sh
$ cd tools
$ make -j
```
or using local [miniconda](https://conda.io/docs/glossary.html#miniconda-glossary)
```sh
$ cd tools
$ make -f conda.mk -j
```


To use cuda (and cudnn), make sure to set paths in your `.bashrc` or `.bash_profile` appropriately.
```
CUDAROOT=/path/to/cuda

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
```

## Execution of example scripts
Move to an example directory under the `egs` directory.
We prepare several major ASR benchmarks including WSJ, CHiME-4, and TED.
The following directory is an example of performing ASR experiment with the VoxForge Italian Corpus.
```sh
$ cd egs/voxforge/asr1
```
Once move to the directory, then, execute the following main script with a **chainer** backend:
```sh
$ ./run.sh
```
or execute the following main script with a **pytorch** backend 
(currently the pytorch backend does not support VGG-like layers):
```sh
$ ./run.sh --backend pytorch --etype blstmp
```
With this main script, you can perform a full procedure of ASR experiments including
- Data download
- Data preparation (Kaldi style, see http://kaldi-asr.org/doc/data_prep.html)
- Feature extraction (Kaldi style, see http://kaldi-asr.org/doc/feat.html)
- Dictionary and JSON format data preparation
- Training based on [chainer](https://chainer.org/) or [pytorch](http://pytorch.org/).
- Recognition and scoring

### Use of GPU
If you use GPU in your experiment, set `--gpu` option in `run.sh` appropriately, e.g., 
```sh
$ ./run.sh --gpu 0
```
Default setup uses CPU (`--gpu -1`).

### Setup in your cluster
Change `cmd.sh` according to your cluster setup.
If you run experiments with your local machine, please use default `cmd.sh`.
For more information about `cmd.sh` see http://kaldi-asr.org/doc/queue.html.
It supports Grid Engine (`queue.pl`), SLURM (`slurm.pl`), etc.

### Error due to matplotlib
If you have the following error (or other numpy related errors), 
```
RuntimeError: module compiled against API version 0xc but this version of numpy is 0xb
Exception in main training loop: numpy.core.multiarray failed to import
Traceback (most recent call last):
;
:
from . import _path, rcParams
ImportError: numpy.core.multiarray failed to import
```
Then, please reinstall matplotlib with the following command:
```sh
$ cd egs/voxforge/asr1
$ . ./path.sh
$ pip install pip --upgrade; pip uninstall matplotlib; pip --no-cache-dir install matplotlib
```

## Installation using Docker

For GPU support nvidia-docker should be installed.

For Execution use the command 
```sh
$ cd egs/voxforge/asr1
$ ./run_in_docker.sh --gpu GPUID
```

If GPUID is set to -1, the program will run only CPU.

The file builds and loads the information into the Docker container. If any additional application is required, modify the Docker devel-file located at the tools folder.

To downgrade or use a private devel file, modify the name inside run_in_docker.sh

## Results

We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

|           | CER (%) | WER (%)  |
|-----------|:----:|:----:|
| WSJ dev93 |  5.5 | 13.1 |
| WSJ eval92|  3.8 |  9.3 |
| CSJ eval1 | 9.7 | N/A  |
| CSJ eval2 |  6.9 | N/A  |
| CSJ eval3 |  7.5 | N/A  |
| HKUST train_dev | 29.7 | N/A  |
| HKUST dev       | 28.3 | N/A  |
| Librispeech dev_clean  | 2.9 | 7.7 |
| Librispeech test_clean | 2.7 | 7.7 |


## References (Please cite the following articles)
[1] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[2] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

