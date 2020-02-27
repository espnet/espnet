## Usage
### Directory structure

```
espnet/              # Python modules
utils/               # Utility scripts of ESPnet
test/                # Unit test
test_utils/          #unit test for executable scripts
egs/                 # The complete recipe for each corpora
    an4/             # AN4 is tiny corpus and can be obtained freely, so it might be suitable for tutorial
      asr1/          # ASR recipe
          - run.sh   # Executable script
          - cmd.sh   # To select the backend for job scheduler 
          - path.sh  # Setup script for environment variables
          - conf/    # Containing COnfiguration files
          - steps/   # The utils scripts from Kaldi
          - utils/   # The utils scripts from Kaldi
      tts1/          # TTS recipe
    ...
```

### Execution of example scripts

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
- [Data preparation](http://kaldi-asr.org/doc/data_prep.html) (Kaldi style)
- [Feature extraction](http://kaldi-asr.org/doc/feat.html) (Kaldi style)
- Dictionary and JSON format data preparation
- Training based on [chainer](https://chainer.org/) or [pytorch](http://pytorch.org/).
- Recognition and scoring

### Setup in your cluster
Change `cmd.sh` according to your cluster setup.
If you run experiments with your local machine, you don't have to change it.
For more information about `cmd.sh` see [Parallelization in Kaldi
](http://kaldi-asr.org/doc/queue.html).
It supports Grid Engine (`queue.pl`), SLURM (`slurm.pl`), etc.

You also changing the configuration to use specified backend:


|cmd     |Backend                                  | configuration file|
|--------| :--------------------------------------:| :---------------: |
|run.pl  | Local machine (default)                 |-                  |
|queue.pl|Sun grid engine, or grid endine like tool|conf/queue.conf    |
|slurm.pl|Slurm                                    |conf/queue.conf    |
|pbs.pl  |PBS/Torque                               |conf/pbs.conf      |
|ssh.pl  |SSH                                      |.queue/machines    |


### Logging

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


### Change options in run.sh

We rely on [utils/parse_options.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh) to paser command line arguments in shell script and it's used in run.sh: 

e.g. If the script has `ngpu` option

```bash
#!/bin/bash
# run.sh
ngpu=1
. utils/parse_options.sh
echo ${ngpu}
```

Then you can change the value as following:

```bash
$ ./run.sh --ngpu 2
echo 2
```

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


### Start from the middle stage or stop at specified stage

`run.sh` has multiple stages including data prepration, traning, and etc., so you may likely want to start 
from the specified stage if some stages are failed by some reason for example.

You can start from specified stage as following and stop the process at the specifed stage:

```bash
# Start from 3rd stage and stop at 5th stage
$ ./run.sh --stage 3 --stop-stage 5
```


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


### Changing the training configuration

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



### Known issues

#### Error due to ACS (Multiple GPUs)

When using multiple GPUs, if the training freezes or lower performance than expected is observed, verify that PCI Express Access Control Services (ACS) are disabled.
Larger discussions can be found at: [link1](https://devtalk.nvidia.com/default/topic/883054/multi-gpu-peer-to-peer-access-failing-on-tesla-k80-/?offset=26) [link2](https://www.linuxquestions.org/questions/linux-newbie-8/howto-list-all-users-in-system-380426/) [link3](https://github.com/pytorch/pytorch/issues/1637).
To disable the PCI Express ACS follow instructions written [here](https://github.com/NVIDIA/caffe/issues/10). You need to have a ROOT user access or request to your administrator for it.

#### Error due to matplotlib

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


### Chainer and Pytorch backends

|                    | Chainer                         | Pytorch                            |
| -----------        | :----:                          | :----:                             |
| Performance        | ◎                               | ◎                                  |
| Speed              | ○                               | ◎                                  |
| Multi-GPU          | supported                       | supported                          |
| VGG-like encoder   | supported                       | supported                          |
| Transformer        | supported                       | supported                          |
| RNNLM integration  | supported                       | supported                          |
| #Attention types   | 3 (no attention, dot, location) | 12 including variants of multihead |
| TTS recipe support | no support                      | supported                          |
