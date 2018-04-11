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
- Attention: Dot product, location-aware attention, variants of multihead (pytorch only)
- Incorporate RNNLM/LSTMLM trained only with text data
- Flexible network architecture thanks to chainer and pytorch
- Kaldi style complete recipe 
  - Support numbers of ASR benchmarks (WSJ, Switchboard, CHiME-4, Librispeech, TED, CSJ, AMI, HKUST, Voxforge, etc.)
- State-of-the-art performance in Japanese/Chinese benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- Moderate performance in standard English benchmarks

## Requirements
- Python2.7+  
- Cuda 8.0 (for the use of GPU)  
- Cudnn 6 (for the use of GPU)  
- NCCL 2.0+ (for the use of multi-GPUs)

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

If you want to use multiple GPUs, you should install [nccl](https://developer.nvidia.com/nccl) 
and set paths in your `.bashrc` or `.bash_profile` appropriately, for 
```
CUDAROOT=/path/to/cuda
NCCL_ROOT=/path/to/nccl

export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$CUDAROOT/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH
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
If you use GPU in your experiment, set `--ngpu` option in `run.sh` appropriately, e.g., 
```sh
# use single gpu
$ ./run.sh --ngpu 1

# use multi-gpu
$ ./run.sh --ngpu 3

# use cpu
$ ./run.sh --ngpu 0
```
Default setup uses CPU (`--ngpu 0`).  

Note that if you want to use multi-gpu, the installation of [nccl](https://developer.nvidia.com/nccl) 
is required before setup.

### Docker Container
To work inside a docker container, execute `run.sh` located inside the docker directory.
It will build a container and execute the main program specified by the following GPU, ASR example, and outside directory information, as follows:
```sh
$ cd docker
$ ./run.sh [--docker_gpu 0 --docker_egs chime4 --docker_folders /export/corpora4/CHiME4/CHiME3] --dlayers 1 --ngpu 1 
```
The arguments required for the docker configuration have a prefix "--docker" (e.g., `--docker_gpu`, `--docker_egs`, `--docker_folders`). `run.sh` accept all normal ESPnet arguments, which must be followed by these docker arguments.
Multiple GPUs should be specified with the following options:
```sh
$ cd docker
$ ./run.sh --docker_gpu 0,1,2 --docker_egs chime5 --docker_folders /export/corpora4/CHiME5 --ngpu 3
```
Note that all experimental files and results are created under the normal example directories (`egs/<example>/asr1/`).

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

## Results

We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

|           | CER (%) | WER (%)  |
|-----------|:----:|:----:|
| WSJ dev93 | 5.3 | 12.4 |
| WSJ eval92| 3.6 |  8.9 |
| CSJ eval1 | 8.5 | N/A  |
| CSJ eval2 | 6.1 | N/A  |
| CSJ eval3 | 6.8 | N/A  |
| HKUST train_dev | 29.7 | N/A  |
| HKUST dev       | 28.3 | N/A  |
| Librispeech dev_clean  | 2.9 | 7.7 |
| Librispeech test_clean | 2.7 | 7.7 |


## Chainer and Pytorch backends

|           | Chainer | Pytorch |
|-----------|:----:|:----:|
| Performance | ◎ | ○ |
| Speed | ○ | ◎ |
| Multi-GPU | supported | no support |
| VGG-like encoder | supported | no support |
| RNNLM integration | supported | supported |
| #Attention types | 3 (no attention, dot, location) | 12 including variants of multihead |

## References (Please cite the following articles)
[1] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[2] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

