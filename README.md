# ESPnet: end-to-end speech processing toolkit

[![Build Status](https://travis-ci.org/espnet/espnet.svg?branch=master)](https://travis-ci.org/espnet/espnet)

ESPnet is an end-to-end speech processing toolkit, mainly focuses on end-to-end speech recognition, and end-to-end text-to-speech.
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
- Tacotron2 based end-to-end TTS (new!)

## Requirements

- Python2.7+  
- Cuda 8.0 or 9.1 (for the use of GPU)  
- Cudnn 6+ (for the use of GPU)  
- NCCL 2.0+ (for the use of multi-GPUs)
- protocol buffer (for the sentencepiece, you need to install via package manager e.g. `sudo apt-get install libprotobuf9v5 protobuf-compiler libprotobuf-dev`. See details `Installation` of https://github.com/google/sentencepiece/blob/master/README.md)

- PyTorch 0.4.1+
- Chainer 4.3.1

## Installation

### Step 1) setting of the environment

To use cuda (and cudnn), make sure to set paths in your `.bashrc` or `.bash_profile` appropriately.
```
CUDAROOT=/path/to/cuda

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
```

If you want to use multiple GPUs, you should install [nccl](https://developer.nvidia.com/nccl) 
and set paths in your `.bashrc` or `.bash_profile` appropriately, for example:
```
CUDAROOT=/path/to/cuda
NCCL_ROOT=/path/to/nccl

export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$CUDAROOT/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
```

### Step 2-A) installation with compiled Kaldi

Install Python libraries and other required tools using system python and virtualenv
```sh
$ cd tools
$ make KALDI=/path/to/kaldi
```
or using local [miniconda](https://conda.io/docs/glossary.html#miniconda-glossary)
```sh
$ cd tools
$ make KALDI=/path/to/kaldi -f conda.mk
```

### Step 2-B) installation including Kaldi installation

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

### Step 2-C) installation with specified python

Install Kaldi, Python libraries and other required tools using specified python and virtualenv
```sh
$ cd tools
$ make -j PYTHON=/path/to/python2.7
```
You can also specified `python3.6`, but some preprocessing functions require `python2.7`.  
So we recommend to use `python2.7`.

### Step 3) installation check

You can check whether the install is succeeded via the following commands
```sh
$ cd tools
$ source venv/bin/activate && python check_install.py
```
If you have no warning, ready to run the recipe!

If there are some problems in python libraries, you can re-setup only python environment via following commands
```sh
$ cd tools
$ make clean_python
$ make all_python
```
And then check the install is succeeded again.

## Execution of example scripts

Move to an example directory under the `egs` directory.
We prepare several major ASR benchmarks including WSJ, CHiME-4, and TED.
The following directory is an example of performing ASR experiment with the CMU Census Database (AN4) recipe.
```sh
$ cd egs/an4/asr1
```
Once move to the directory, then, execute the following main script with a **chainer** backend:
```sh
$ ./run.sh --backend chainer
```
or execute the following main script with a **pytorch** backend:
```sh
$ ./run.sh --backend pytorch
```
With this main script, you can perform a full procedure of ASR experiments including
- Data download
- Data preparation (Kaldi style, see http://kaldi-asr.org/doc/data_prep.html)
- Feature extraction (Kaldi style, see http://kaldi-asr.org/doc/feat.html)
- Dictionary and JSON format data preparation
- Training based on [chainer](https://chainer.org/) or [pytorch](http://pytorch.org/).
- Recognition and scoring

The training progress (loss and accuracy for training and validation data) can be monitored with the following command
```
$ tail -f exp/${expdir}/train.log
```
With the default verbose (=0), it gives you the following information
```
epoch       iteration   main/loss   main/loss_ctc  main/loss_att  validation/main/loss  validation/main/loss_ctc  validation/main/loss_att  main/acc    validation/main/acc  elapsed_time  eps
:
:
6           89700       63.7861     83.8041        43.768                                                                                   0.731425                         136184        1e-08
6           89800       71.5186     93.9897        49.0475                                                                                  0.72843                          136320        1e-08
6           89900       72.1616     94.3773        49.9459                                                                                  0.730052                         136473        1e-08
7           90000       64.2985     84.4583        44.1386        72.506                94.9823                   50.0296                   0.740617    0.72476              137936        1e-08
7           90100       81.6931     106.74         56.6462                                                                                  0.733486                         138049        1e-08
7           90200       74.6084     97.5268        51.6901                                                                                  0.731593                         138175        1e-08
     total [#################.................................] 35.54%
this epoch [#####.............................................] 10.84%
     91300 iter, 7 epoch / 20 epochs
   0.71428 iters/sec. Estimated time to finish: 2 days, 16:23:34.613215.
```

### Use of GPU

If you use GPU in your experiment, set `--ngpu` option in `run.sh` appropriately, e.g., 
```sh
# use single gpu
$ ./run.sh --ngpu 1

# use multi-gpu
$ ./run.sh --ngpu 3

# if you want to specify gpus, set CUDA_VISIBLE_DEVICES as follows
# (Note that if you use slurm, this specification is not needed)
$ CUDA_VISIBLE_DEVICES=0,1,2 ./run.sh --ngpu 3

# use cpu
$ ./run.sh --ngpu 0
```
Default setup uses CPU (`--ngpu 0`).

Note that if you want to use multi-gpu, the installation of [nccl](https://developer.nvidia.com/nccl) 
is required before setup.

### Error due to ACS (Multiple GPUs)

When using multiple GPUs, if the training freezes or lower performance than expected is observed, verify that PCI Express Access Control Services (ACS) are disabled.
Larger discussions can be found at: [link1](https://devtalk.nvidia.com/default/topic/883054/multi-gpu-peer-to-peer-access-failing-on-tesla-k80-/?offset=26) [link2](https://www.linuxquestions.org/questions/linux-newbie-8/howto-list-all-users-in-system-380426/) [link3](https://github.com/pytorch/pytorch/issues/1637).
To disable the PCI Express ACS follow instructions written [here](https://github.com/NVIDIA/caffe/issues/10). You need to have a ROOT user access or request to your administrator for it.

### Docker Container

go to docker/ and follow [README.md](https://github.com/espnet/espnet/tree/master/docker/README.md) instructions there.

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
$ cd egs/an4/asr1
$ . ./path.sh
$ pip install pip --upgrade; pip uninstall matplotlib; pip --no-cache-dir install matplotlib
```

## CTC, attention, and hybrid CTC/attention

ESPnet can completely switch the mode from CTC, attention, and hybrid CTC/attention

```sh
# hybrid CTC/attention (default)
#  --mtlalpha 0.5 and --ctc_weight 0.3 in most cases
$ ./run.sh

# CTC mode
$ ./run.sh --mtlalpha 1.0 --ctc_weight 1.0 --recog_model model.loss.best

# attention mode
$ ./run.sh --mtlalpha 0.0 --ctc_weight 0.0
```

The CTC training mode does not output the validation accuracy, and the optimum model is selected with its loss value 
(i.e., `--recog_model model.loss.best`).
About the effectiveness of the hybrid CTC/attention during training and recognition, see [1] and [2].

## Results

We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

|           | CER (%) | WER (%)  |
|-----------|:----:|:----:|
| WSJ dev93 | 4.9 | 10.9 |
| WSJ eval92| 3.1 |  7.1 |
| CSJ eval1 | 7.3 | N/A  |
| CSJ eval2 | 5.3 | N/A  |
| CSJ eval3 | 5.9 | N/A  |
| HKUST train_dev | 28.8 | N/A  |
| HKUST dev       | 27.4 | N/A  |
| Librispeech dev_clean  | N/A | 4.7 |
| Librispeech test_clean | N/A | 4.7 |

Note that the performance of the CSJ, HKUST, and Librispeech tasks was significantly improved by using the wide network (#units = 1024) and large subword units if necessary reported by [RWTH](https://arxiv.org/pdf/1805.03294.pdf).

## Chainer and Pytorch backends

|           | Chainer | Pytorch |
|-----------|:----:|:----:|
| Performance | ◎ | ◎ |
| Speed | ○ | ◎ |
| Multi-GPU | supported | supported |
| VGG-like encoder | supported | supported |
| RNNLM integration | supported | supported |
| #Attention types | 3 (no attention, dot, location) | 12 including variants of multihead |
| TTS recipe support | no support | supported |

## References (Please cite the following articles)

[1] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[2] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

