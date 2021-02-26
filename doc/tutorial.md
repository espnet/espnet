## Usage
### Directory structure

```
espnet/              # Python modules
utils/               # Utility scripts of ESPnet
test/                # Unit test
test_utils/          # Unit test for executable scripts
egs/                 # The complete recipe for each corpora
    an4/             # AN4 is tiny corpus and can be obtained freely, so it might be suitable for tutorial
      asr1/          # ASR recipe
          - run.sh   # Executable script
          - cmd.sh   # To select the backend for job scheduler
          - path.sh  # Setup script for environment variables
          - conf/    # Containing Configuration files
          - steps/   # The steps scripts from Kaldi
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
See [Using Job scheduling system](./parallelization.md)

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

### Multiple GPU TIPs
- Note that if you want to use multiple GPUs, the installation of [nccl](https://developer.nvidia.com/nccl) is required before setup.
- Currently, espnet1 only supports multiple GPU training within a single node. The distributed setup across multiple nodes is only supported in [espnet2](https://espnet.github.io/espnet/espnet2_distributed.html). 
- We don't support multiple GPU inference. Instead, please split the recognition task for multiple jobs and distribute these split jobs to multiple GPUs.
- If you could not get enough speed improvement with multiple GPUs, you should first check the GPU usage by `nvidia-smi`. If the GPU-Util percentage is low, the bottleneck would come from the disk access. You can apply data prefetching by `--n-iter-processes 2` in your `run.sh` to mitigate the problem. Note that this data prefetching consumes a lot of CPU memory, so please be careful when you increase the number of processes.

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
$ ./run.sh --mtlalpha 0.0 --ctc_weight 0.0 --maxlenratio 0.8 --minlenratio 0.3
```

- The CTC training mode does not output the validation accuracy, and the optimum model is selected with its loss value
(i.e., `--recog_model model.loss.best`).
- The pure attention mode requires to set the maximum and minimum hypothesis length (`--maxlenratio` and `--minlenratio`), appropriately. In general, if you have more insertion errors, you can decrease the `maxlenratio` value, while if you have more deletion errors you can increase the `minlenratio` value. Note that the optimum values depend on the ratio of the input frame and output label lengths, which is changed for each language and each BPE unit.
- About the effectiveness of hybrid CTC/attention during training and recognition, see [2] and [3]. For example, hybrid CTC/attention is not sensitive to the above maximum and minimum hypothesis heuristics. 

### Transducer

ESPnet also supports transducer-based models.
To switch to transducer mode, the following should be set in the training config:

```
criterion: loss
model-module: "espnet.nets.pytorch_backend.e2e_asr_transducer:E2E"
```

Several transducer architectures are currently available:
- RNN-Transducer (default)
- Custom-Transducer (`etype: custom` and `dtype: custom`)
- Mixed Custom/RNN-Transducer (e.g: `etype: custom` with `dtype: lstm`)

The architecture specification is separated for the encoder and decoder parts, and defined by the user through, respectively, `etype` and `dtype` in training config. If `custom` is specified for either, a customizable architecture will be used for the corresponding part, otherwise a RNN-based architecture will be selected.

While defining a RNN architecture is done in an usual manner (similarly to CTC, Att and MTL) with global parameters, a customizable architecture definition for transducer is different:
1) Each blocks (or layers) for both network part should be specified individually through `enc-block-arch` or/and `dec-block-arch`:

        # e.g: TDNN-Transformer encoder
        etype: custom
        enc-block-arch:
                - type: tdnn
                  idim: 512
                  odim: 320
                  ctx_size: 3
                  dilation: 1
                  stride: 1
                - type: transformer
                  d_hidden: 320
                  d_ff: 320
                  heads: 4

2) Each part has different allowed block type: `tdnn`, `conformer` or `transformer` for encoder and `causal-conv1d` or `transformer` for decoder. For each block type, a set of parameters are needed:

        # TDNN
        - type: tdnn
          idim: input dimension
          odim: output dimension
          ctx_size: size of the context window
          dilation: parameter to control the stride of elements within the neighborhood
          stride: stride of the sliding blocks
          [optional: dropout-rate]

        # Transformer
        - type: transformer
          d_hidden: input/output dimension
          d_ff: feed-forward hidden dimension
          heads: number of heads in multi-head attention
          [optional: dropout-rate, pos-dropout-rate, att-dropout-rate]

        # Conformer
        - type: conformer
          d_hidden: input/output dimension
          d_ff: feed-forward hidden dimension
          heads: number of heads in multi-head attention
          macaron_style: wheter to use macaron style
          use_conv_mod: whether to use convolutional module
          conv_mod_kernel: number of kernel in convolutional module (optional if `use_conv_mod=True`)
          [optional: dropout-rate, pos-dropout-rate, att-dropout-rate]

        # Causal Conv1d
        - type: causal-conv1d
          idim: input dimension
          odim: output dimension
          kernel_size: size of convolving kernel
          stride: stride of the convolution
          dilation: spacing between the kernel points

3) Each specified block(s) for each network part can be repeated by specifying the number of duplications through `enc-block-repeat` or `dec-block-repeat` parameters:

        # e.g.: 2x (Causal-Conv1d + Transformer) decoder
        dtype: transformer
        dec-block-arch:
                - type: causal-conv1d
                  idim: 256
                  odim: 256
                  kernel_size: 5
                - type: transformer
                  d_hidden: 256
                  d_ff: 256
                  heads: 4
                  dropout-rate: 0.1
                  att-dropout-rate: 0.4
        dec-block-repeat: 2

For more information about the customizable architecture, please refer to [vivos config examples](https://github.com/espnet/espnet/tree/master/egs/vivos/asr1/conf/tuning/transducer) which cover all cases.

Various decoding algorithms are also available for transducer by setting `search-type` parameter in decode config:
- Default beam search (`default`)
- Time-synchronous decoding (`tsd`)
- Alignment-length decoding (`alsd`)
- N-step Constrained beam search (`nsc`)

All algorithms share a common parameter to control beam size (`beam-size`) but each ones have its own parameters:

        # Default beam search
        search-type: default
        score-norm-transducer: normalize final scores by length

        # Time-synchronous decoding
        search-type: tsd
        max-sym-exp: number of maximum symbol expansions at each time step

        # Alignement-length decoding
        search-type: alsd
        u-max: maximum output sequence length

        # N-step Constrained beam search
        search-type: nsc
        nstep: number of maximum expansion steps at each time step
               (N exp. step = N symbol expansion + 1)
        prefix-alpha: maximum prefix length in prefix search

Except for the default algorithm, performance and decoding time can be controlled through described parameters. A high value will increase performance but also decoding time while a low value will decrease decoding time but will negatively impact performance.

IMPORTANT (temporary) note: ALSD, TSD and NSC have their execution time degraded because of the current batching implementation. We decided to keep it as if for internal discussions but it can be manually removed by the user to speed up inference. In a near future, the inference part for transducer will be replaced by our own torch lib.

The algorithm references can be found in [methods documentation](https://github.com/espnet/espnet/tree/master/espnet/nets/beam_search_transducer.py). For more information about decoding usage, refer to [vivos config examples](https://github.com/espnet/espnet/tree/master/egs/vivos/asr1/conf/tuning/transducer).

Additional notes:
- Similarly to CTC training mode, transducer does not output the validation accuracy. Thus, the optimum model is selected with its loss value (i.e., --recog_model model.loss.best).
- There are several differences between MTL and transducer training/decoding options. The users should refer to `espnet/espnet/nets/pytorch_backend/e2e_asr_transducer.py` for an overview.
- RNN-decoder pre-initialization using a LM is supported. The LM state dict keys (`predictor.*`) will be matched to AM state dict keys (`dec.*`).
- Transformer-decoder pre-initialization using a transformer LM is not supported yet.
- Transformer and conformer blocks within the same architecture part (i.e: encoder) is not supported yet.
- Customizable architecture is a in-progress work and will be eventually extended to RNN. Please report any encountered error or usage issue.

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

### How to use finetuning

ESPnet currently supports two finetuning operations: transfer learning (1.x) and freezing (2.).

1.1. Transfer learning option is split between encoder initialization (`--enc-init`) and decoder initialization (`--dec-init`). However, the same model can be specified for both options. Each option takes a snapshot path (e.g.: `exp/[model]/results/snapshot.ep.1`) or model path (e.g.: `exp/[model]/results/model.loss.best`) as argument.

1.2. Additionally, a list of modules (separated by a comma) can be specified to control the modules to transfer using `--enc-init-mods` and `--dec-init-mods` options.
It should be noted the user doesn't need to specify each module individually, only a partial matching (beginning of the string) is needed.

Example 1: `--enc-init-mods='enc.'` means all encoder modules should be transfered.

Example 2: `--enc-init-mods='enc.embed.,enc.0.'` means encoder embedding layer and first layer should be transfered.

2. Freezing option can be used through `--freeze-mods`. Similarly to `--(enc|dec)-init-mods`, the option take a list of modules (separated by a comma). The behaviour being the same (partial matching).

Example 1: `--freeze-mods='enc.embed.'` means encoder embedding layer should be frozen.

Example 2: `--freeze-mods='dec.embed,dec.0.'` means decoder embedding layer and first layer should be frozen.

3. RNN-based and Transformer-based models have different key names for encoder and decoder parts:
 - RNN model has `enc` for encoder and `dec` for decoder.
 - Transformer has `encoder` for encoder and `decoder` for decoder.

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
