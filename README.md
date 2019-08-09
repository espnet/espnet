<div align="left"><img src="doc/image/espnet_logo1.png" width="550"/></div>

# ESPnet: end-to-end speech processing toolkit

[![Build Status](https://travis-ci.org/espnet/espnet.svg?branch=master)](https://travis-ci.org/espnet/espnet)
[![CircleCI](https://circleci.com/gh/espnet/espnet.svg?style=svg)](https://circleci.com/gh/espnet/espnet)
[![codecov](https://codecov.io/gh/espnet/espnet/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet)
[![Gitter](https://badges.gitter.im/espnet-en/community.svg)](https://gitter.im/espnet-en/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

ESPnet is an end-to-end speech processing toolkit, mainly focuses on end-to-end speech recognition and end-to-end text-to-speech.
ESPnet uses [chainer](https://chainer.org/) and [pytorch](http://pytorch.org/) as a main deep learning engine,
and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for speech recognition and other speech processing experiments.

* [Key Features](#key-features)
* [Requirements](#requirements)
* [Installation](#installation)
  * [Step 1) setting of the environment for GPU support](#step-1-setting-of-the-environment-for-gpu-support)
  * [Step 2\-A) installation with compiled Kaldi](#step-2-a-installation-with-compiled-kaldi)
    * [using miniconda (default)](#using-miniconda-default)
    * [using existing python](#using-existing-python)
  * [Step 2\-B) installation including Kaldi installation](#step-2-b-installation-including-kaldi-installation)
  * [Step 2\-C) installation for CPU\-only](#step-2-c-installation-for-cpu-only)
  * [Step 3) installation check](#step-3-installation-check)
* [Execution of example scripts](#execution-of-example-scripts)
  * [Use of GPU](#use-of-gpu)
  * [Changing the configuration](#changing-the-configuration)
  * [How to set minibatch](#how-to-set-minibatch)
  * [Setup in your cluster](#setup-in-your-cluster)
  * [CTC, attention, and hybrid CTC/attention](#ctc-attention-and-hybrid-ctcattention)
* [Known issues](#known-issues)
  * [Error due to ACS (Multiple GPUs)](#error-due-to-acs-multiple-gpus)
  * [Error due to matplotlib](#error-due-to-matplotlib)
* [Docker Container](#docker-container)
* [Results and demo](#results-and-demo)
  * [ASR results](#asr-results)
  * [ASR demo](#asr-demo)
  * [TTS results](#tts-results)
  * [TTS demo](#tts-demo)
* [Chainer and Pytorch backends](#chainer-and-pytorch-backends)
* [References](#references)
* [Citation](#citation)

## Key Features
- Hybrid CTC/attention based end-to-end ASR
  - Fast/accurate training with CTC/attention multitask training
  - CTC/attention joint decoding to boost monotonic alignment decoding
- Encoder: VGG-like CNN + BiRNN (LSTM/GRU), sub-sampling BiRNN (LSTM/GRU) or Transformer
- Attention: Dot product, location-aware attention, variants of multihead
- Incorporate RNNLM/LSTMLM trained only with text data
- Batch GPU decoding
- Tacotron2 based end-to-end TTS
- Transformer based end-to-end TTS (new!)
- Flexible network architecture thanks to chainer and pytorch
- Kaldi style complete recipe
  - Support numbers of ASR recipes (WSJ, Switchboard, CHiME-4/5, Librispeech, TED, CSJ, AMI, HKUST, Voxforge, REVERB, etc.)
  - Support numbers of TTS recipes with a similar manner to the ASR recipe (LJSpeech, LibriTTS, M-AILABS, etc.)
  - Support speech translation recipes (Fisher callhome Spanish to English, IWSLT'18)
  - Support speech separation and recognition recipe (WSJ-2mix)
- State-of-the-art performance in several benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- Flexible front-end processing thanks to [kaldiio](https://github.com/nttcslab-sp/kaldiio) and HDF5 support
- Tensorboard based monitoring

## Requirements

- Python 3.6.1+
- protocol buffer (for the sentencepiece, you need to install via package manager e.g. `sudo apt-get install libprotobuf9v5 protobuf-compiler libprotobuf-dev`. See details `Installation` of https://github.com/google/sentencepiece/blob/master/README.md)

- PyTorch 0.4.1, 1.0.0, 1.0.1
- gcc 4.9+ for PyTorch1.0.0+
- Chainer 6.0.0

Optionally, GPU environment requires the following libraries:

- Cuda 8.0, 9.0, 9.1, 10.0 depending on each DNN library
- Cudnn 6+
- NCCL 2.0+ (for the use of multi-GPUs)

## Installation

### Step 1) setting of the environment for GPU support

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

#### using miniconda (default)

Install Python libraries and other required tools with [miniconda](https://conda.io/docs/glossary.html#miniconda-glossary)
```sh
$ cd tools
$ make KALDI=/path/to/kaldi
```

You can also specify the Python (`PYTHON_VERSION` default 3.7), PyTorch (`TH_VERSION` default 1.0.0) and CUDA versions (`CUDA_VERSION` default 10.0), for example:
```sh
$ cd tools
$ make KALDI=/path/to/kaldi PYTHON_VERSION=3.6 TH_VERSION=0.4.1 CUDA_VERSION=9.0
```

#### using existing python

If you do not want to use miniconda, you need to specify your python interpreter to setup `virtualenv`

```sh
$ cd tools
$ make KALDI=/path/to/kaldi PYTHON=/usr/bin/python3.6
```

### Step 2-B) installation including Kaldi installation

Install Kaldi, Python libraries and other required tools with [miniconda](https://conda.io/docs/glossary.html#miniconda-glossary)
```sh
$ cd tools
$ make -j 10
```

As seen above, you can also specify the Python and CUDA versions, and Python path (based on `virtualenv`), for example:
```sh
$ cd tools
$ make -j 10 PYTHON_VERSION=3.6 TH_VERSION=0.4.1 CUDA_VERSION=9.0
```
```sh
$ cd tools
$ make -j 10 PYTHON=/usr/bin/python3.6
```


### Step 2-C) installation for CPU-only

To install in a terminal that does not have a GPU installed, just clear the version of `CUPY` as follows:

```sh
$ cd tools
$ make CUPY_VERSION='' -j 10
```

This option is enabled for any of the install configuration.

### Step 3) installation check

You can check whether the install is succeeded via the following commands
```sh
$ cd tools
$ make check_install
```
or `make check_install CUPY_VERSION=''` if you do not have a GPU on your terminal.
If you have no warning, ready to run the recipe!

If there are some problems in python libraries, you can re-setup only python environment via following commands
```sh
$ cd tools
$ make clean_python
$ make python
```

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
```sh
$ tail -f exp/${expdir}/train.log
```
When we use `./run.sh --verbose 0` (`--verbose 0` is default in most recipes), it gives you the following information
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
Note that the an4 recipe uses `--verbose 1` as default since this recipe is often used for a debugging purpose.

In addition [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) events are automatically logged in the `tensorboard/${expname}` folder. Therefore, when you install Tensorboard, you can easily compare several experiments by using
```sh
$ tensorboard --logdir tensorboard
```
and connecting to the given address (default : localhost:6006). This will provide the following information:
![2018-12-18_19h49_48](https://user-images.githubusercontent.com/14289171/50175839-2491e280-02fe-11e9-8dfc-de303804034d.png)
Note that we would not include the installation of Tensorboard to simplify our installation process. Please install it manually (`pip install tensorflow; pip install tensorboard`) when you want to use Tensorboard.

### Use of GPU
- Training:
  If you want to use GPUs in your experiment, please set `--ngpu` option in `run.sh` appropriately, e.g.,
  ```bash
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
  - Default setup uses a single GPU (`--ngpu 1`).
- ASR decoding:
  ESPnet also supports the GPU-based decoding for fast recognition.
  - Please manually remove the following lines in `run.sh`:
    ```bash
    #### use CPU for decoding
    ngpu=0
    ```
  - Set 1 or more values for `--batchsize` option in `asr_recog.py` to enable GPU decoding
  - And execute the script (e.g., `run.sh --stage 5 --ngpu 1`)
  - You'll achieve significant speed improvement by using the GPU decoding
- Note that if you want to use multi-gpu, the installation of [nccl](https://developer.nvidia.com/nccl) is required before setup.

### Changing the configuration
The default configurations for training and decoding are written in `conf/train.yaml` and `conf/decode.yaml` respectively.  It can be overwritten by specific arguments: e.g.

```bash
# e.g.
asr_train.py --config conf/train.yaml --batch-size 24
# e.g.--config2 and --config3 are also provided and the latter option can overwrite the former.
asr_train.py --config conf/train.yaml --config2 conf/new.yaml
```

In this way, you need to edit `run.sh` and it might be inconvenient sometimes.
Instead of giving arguments directly, we recommend you to modify the yaml file and give it to `run.sh`:

```bash
# e.g.
./run.sh --train-config conf/train_modified.yaml
# e.g.
./run.sh --train-config conf/train_modified.yaml --decode-config conf/decode_modified.yaml
```

We also provide a utility to generate a yaml file from the input yaml file:

```bash
# e.g. You can give any parameters as '-a key=value' and '-a' is repeatable. 
#      This generates new file at 'conf/train_batch-size24_epochs10.yaml'
./run.sh --train-config $(change_yaml.py conf/train.yaml -a batch-size=24 -a epochs=10)
# e.g. '-o' option specifies the output file name instead of auto named file.
./run.sh --train-config $(change_yaml.py conf/train.yaml -o conf/train2.yaml -a batch-size=24)
```

### How to set minibatch

From espnet v0.4.0, we have three options in `--batch-count` to specify minibatch size (see `espnet.utils.batchfy` for implementation);
1. `--batch-count seq --batch-seqs 32 --batch-seq-maxlen-in 800 --batch-seq-maxlen-out 150`. 

    This option is compatible to the old setting before v0.4.0. This counts the minibatch size as the number of sequences and reduces the size when the maximum length of the input or output sequences is greater than 800 or 150, respectively.
1. `--batch-count bin --batch-bins 100000`. 

    This creates the minibatch that has the maximum number of bins under 100 in the padded input/output minibatch tensor  (i.e., `max(ilen) * idim + max(olen) * odim`). 
Basically, this option makes training iteration faster than `--batch-count seq`. If you already has the best `--batch-seqs x` config, try `--batch-bins $((x * (mean(ilen) * idim + mean(olen) * odim)))`.
1. `--batch-count frame --batch-frames-in 800 --batch-frames-out 100 --batch-frames-inout 900`. 

    This creates the minibatch that has the maximum number of input, output and input+output frames under 800, 100 and 900, respectively. You can set one of `--batch-frames-xxx` partially. Like `--batch-bins`, this option makes training iteration faster than `--batch-count seq`. If you already has the best `--batch-seqs x` config, try `--batch-frames-in $((x * (mean(ilen) * idim)) --batch-frames-out $((x * mean(olen) * odim))`.


### Setup in your cluster

Change `cmd.sh` according to your cluster setup.
If you run experiments with your local machine, please use default `cmd.sh`.
For more information about `cmd.sh` see http://kaldi-asr.org/doc/queue.html.
It supports Grid Engine (`queue.pl`), SLURM (`slurm.pl`), etc.


### CTC, attention, and hybrid CTC/attention

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
About the effectiveness of the hybrid CTC/attention during training and recognition, see [2] and [3].

## Known issues
### Error due to ACS (Multiple GPUs)

When using multiple GPUs, if the training freezes or lower performance than expected is observed, verify that PCI Express Access Control Services (ACS) are disabled.
Larger discussions can be found at: [link1](https://devtalk.nvidia.com/default/topic/883054/multi-gpu-peer-to-peer-access-failing-on-tesla-k80-/?offset=26) [link2](https://www.linuxquestions.org/questions/linux-newbie-8/howto-list-all-users-in-system-380426/) [link3](https://github.com/pytorch/pytorch/issues/1637).
To disable the PCI Express ACS follow instructions written [here](https://github.com/NVIDIA/caffe/issues/10). You need to have a ROOT user access or request to your administrator for it.

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


## Docker Container

go to docker/ and follow [README.md](https://github.com/espnet/espnet/tree/master/docker/README.md) instructions there.


## Results and demo

### ASR results

We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

|           | CER (%) | WER (%)  | Pretrained model |
|-----------|:----:|:----:|:----:|
| Aishell dev            |  6.0 |  N/A | [link](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#transformer-result-default-transformer-with-initial-learning-rate--10-and-epochs--50) |
| Aishell test           |  6.7 |  N/A | same as above |
| CSJ eval1              |  5.7 |  N/A | N/A |
| CSJ eval2              |  4.1 |  N/A | N/A |
| CSJ eval3              |  4.5 |  N/A | N/A |
| HKUST dev              | 23.5 |  N/A | [link](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1/RESULTS.md#transformer-only-20-epochs) |
| Librispeech dev_clean  |  N/A |  2.2 | [link](https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--large-lm) |
| Librispeech dev_other  |  N/A |  5.6 | same as above |
| Librispeech test_clean |  N/A |  2.6 | same as above |
| Librispeech test_other |  N/A |  5.7 | same as above |
| TEDLIUM2 dev           |  N/A | 12.2 | [link](https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md#transformer-default) |
| TEDLIUM2 test          |  N/A | 10.4 | same as above |
| WSJ dev93              |  3.2 |  7.0 | N/A |
| WSJ eval92             |  2.1 |  4.7 | N/A |
| Common Voice dev       |  1.7 |  2.2 | [link](https://github.com/espnet/espnet/blob/master/egs/commonvoice/asr1/RESULTS.md#first-results-default-pytorch-transformer-setting-with-bpe-100-epochs-single-gpu) |
| Common Voice test      |  1.8 |  2.3 | same as above |

Note that the performance of the CSJ, HKUST, and Librispeech tasks was significantly improved by using the wide network (#units = 1024) and large subword units if necessary reported by [RWTH](https://arxiv.org/pdf/1805.03294.pdf).

### ASR demo

You can recognize speech in a WAV file using pretrained models.
Go to a recipe directory and run `utils/recog_wav.sh` as follows:
```sh
cd egs/tedlium2/asr1
../../../utils/recog_wav.sh --models tedlium2.tacotron2.v1 example.wav
```
where `example.wav` is a WAV file to be recognized.
The sampling rate must be consistent with that of data used in training.

Available pretrained models are listed as below.

| Model | Notes |
|:------|:------|
| [tedlium2.tacotron2.v1](https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe) | Streaming decoding based on CTC-based VAD with uni-directional encoder-decoder |

### TTS results

You can access the samples of TTS recipes from following links:

- [Single English speaker Tacotron2](https://drive.google.com/open?id=18JgsOCWiP_JkhONasTplnHS7yaF_konr)
- [Single Japanese speaker Tacotron2](https://drive.google.com/open?id=1fEgS4-K4dtgVxwI4Pr7uOA1h4PE-zN7f)
- [Single other language speaker Tacotron2](https://drive.google.com/open?id=1q_66kyxVZGU99g8Xb5a0Q8yZ1YVm2tN0)
- [Multi English speaker Tacotron2](https://drive.google.com/open?id=1_fKnxuFlLBFCATCsacxKzIy6UBbUPzd0)
- [Single English speaker Transformer (New!)](https://drive.google.com/open?id=14EboYVsMVcAq__dFP1p6lyoZtdobIL1X)

Note that all of the samples uses Griffin-Lim Algorithm to convert wav. Not yet applied neural vocoders.

### TTS demo

You can synthesize speech in a TXT file using pretrained models.
Go to a recipe directory and run `utils/synth_wav.sh` as follows:
```sh
cd egs/ljspeech/tts1
echo "THIS IS A DEMONSTRATION OF TEXT TO SPEECH." > example.txt
../../../utils/synth_wav.sh --models ljspeech.tacotron2.v1 example.txt
```

Available pretrained models are listed as follows.

| Model | Notes |
|:------|:------|
| [libritts.tacotron2.v1](https://drive.google.com/open?id=1iAXwC0AuWusa9AcFeUVkcNLG0I-hnSr3) | Location sensitive attention |
| [ljspeech.tacotron2.v1](https://drive.google.com/open?id=1dKzdaDpOkpx7kWZnvrvx2De7eZEdPHZs) | Location sensitive attention |
| [ljspeech.tacotron2.v2](https://drive.google.com/open?id=11T9qw8rJlYzUdXvFjkjQjYrp3iGfQ15h) | Forward attention |
| [ljspeech.tacotron2.v3](https://drive.google.com/open?id=1hiZn14ITUDM1nkn-GkaN_M3oaTOUcn1n) | Location sensitive attention + guided attention loss |
| [ljspeech.transformer.v1](https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS) | Deep decoder network without reduction factor |
| [ljspeech.transformer.v2](https://drive.google.com/open?id=1xxAwPuUph23RnlC5gym7qDM02ZCW9Unp) | Shallow decoder network with reduction factor = 3 |


## Chainer and Pytorch backends

|           | Chainer | Pytorch |
|-----------|:----:|:----:|
| Performance | ◎ | ◎ |
| Speed | ○ | ◎ |
| Multi-GPU | supported | supported |
| VGG-like encoder | supported | supported |
| Transformer | supported | supported |
| RNNLM integration | supported | supported |
| #Attention types | 3 (no attention, dot, location) | 12 including variants of multihead |
| TTS recipe support | no support | supported |

## References
[1] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-End Speech Processing Toolkit," *Proc. Interspeech'18*, pp. 2207-2211 (2018)

[2] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[3] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

## Citation

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={ESPnet: End-to-End Speech Processing Toolkit},
  year=2018,
  booktitle={Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
```
