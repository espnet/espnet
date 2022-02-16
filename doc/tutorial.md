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
#!/usr/bin/env bash
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

You can start from specified stage as following and stop the process at the specified stage:

```bash
# Start from 3rd stage and stop at 5th stage
$ ./run.sh --stage 3 --stop-stage 5
```


### CTC, attention, and hybrid CTC/attention

ESPnet can easily switch the model's training/decoding mode from CTC, attention, and hybrid CTC/attention.

Each mode can be trained by specifying `mtlalpha` in the [training configuration](https://github.com/espnet/espnet/blob/7dc9da2f07c54b4b0e878d8ef219fcd4d16a5bec/doc/tutorial.md#changing-the-training-configuration):

```sh
# hybrid CTC/attention (default)
mtlalpha: 0.3

# CTC
mtlalpha: 1.0

# attention
mtlalpha: 0.0
```

Decoding for each mode can be done using the following decoding configurations:

```sh
# hybrid CTC/attention (default)
ctc-weight: 0.3
beam-size: 10

# CTC
ctc-weight: 1.0
## for best path decoding
api: v1 # default setting (can be omitted)
## for prefix search decoding w/ beam search
api: v2
beam-size: 10

# attention
ctc-weight: 0.0
beam-size: 10
maxlenratio: 0.8
minlenratio: 0.3
```

- The CTC mode does not compute the validation accuracy, and the optimum model is selected with its loss value
(i.e., `$ ./run.sh --recog_model model.loss.best`).
- The CTC decoding adopts the best path decoding by default, which simply outputs the most probable label at every time step. The prefix search deocding with beam search is also supported in [beam search API v2](https://espnet.github.io/espnet/apis/espnet_bin.html?highlight=api#asr-recog-py).
- The pure attention mode requires to set the maximum and minimum hypothesis length (`--maxlenratio` and `--minlenratio`), appropriately. In general, if you have more insertion errors, you can decrease the `maxlenratio` value, while if you have more deletion errors you can increase the `minlenratio` value. Note that the optimum values depend on the ratio of the input frame and output label lengths, which is changed for each language and each BPE unit. 
- Negative `maxlenratio` can be used to set the constant maximum hypothesis length independently from the number of input frames. If `maxlenratio` is set to `-1`, the decoding will always stop after the first output, which can be used to emulate the utterance classification tasks. This is suitable for some spoken language understanding and speaker identification tasks.
- About the effectiveness of hybrid CTC/attention during training and recognition, see [2] and [3]. For example, hybrid CTC/attention is not sensitive to the above maximum and minimum hypothesis heuristics. 

### Transducer

***Important: If you encounter any issue related to Transducer loss, please open an issue in [our fork of warp-transducer](https://github.com/b-flo/warp-transducer).***

ESPnet supports models trained with Transducer loss, aka Transducer models. To train such model, the following should be set in the training config:

```
criterion: loss
model-module: "espnet.nets.pytorch_backend.e2e_asr_transducer:E2E"
```

#### Architecture

Several Transducer architectures are currently available in ESPnet:
- RNN-Transducer (default, e.g.: `etype: blstm` with `dtype: lstm`)
- Custom-Transducer (e.g.: `etype: custom` and `dtype: custom`)
- Mixed Custom/RNN-Transducer (e.g: `etype: custom` with `dtype: lstm`)

The architecture specification is separated for the encoder and decoder part, and defined by the user through, respectively, `etype` and `dtype` in the training config. If `custom` is specified for either, a customizable architecture will be used for the corresponding part. Otherwise, an RNN-based architecture will be selected.

Here, the *custom* architecture is a unique feature of the Transducer model in ESPnet. It was made available to add some flexibility in the architecture definition and ease the reproduction of some SOTA Transducer models mixing  different layers types or parameters within the same model part (encoder or decoder). As such, the architecture definition is different compared to the RNN architecture :

1) Each block (or layer) of the custom architecture should be specified individually through `enc-block-arch` or/and `dec-block-arch` parameters:

        # e.g: Conv-Transformer encoder
        etype: custom
        enc-block-arch:
                - type: conv1d
                  idim: 80
                  odim: 32
                  kernel_size: [3, 7]
                  stride: [1, 2]
                - type: conv1d
                  idim: 32
                  odim: 32
                  kernel_size: 3
                  stride: 2
                - type: conv1d
                  idim: 32
                  odim: 384
                  kernel_size: 3
                  stride: 1
                - type: transformer
                  d_hidden: 384
                  d_ff: 1536
                  heads: 4

2) Different block types are allowed for the custom encoder (`tdnn`, `conformer` or `transformer`) and the custom decoder (`causal-conv1d` or `transformer`). Each one has a set of mandatory and optional parameters :

        # 1D convolution (TDNN) block
        - type: conv1d
          idim: [Input dimension. (int)]
          odim: [Output dimension. (int)]
          kernel_size: [Size of the context window. (int or tuple)]
          stride (optional): [Stride of the sliding blocks. (int or tuple, default = 1)]
          dilation (optional): [Parameter to control the stride of elements within the neighborhood. (int or tuple, default = 1)]
          groups (optional): [Number of blocked connections from input channels to output channels. (int, default = 1)
          bias (optional): [Whether to add a learnable bias to the output. (bool, default = True)]
          use-relu (optional): [Whether to use a ReLU activation after convolution. (bool, default = True)]
          use-batchnorm: [Whether to use batch normalization after convolution. (bool, default = False)]
          dropout-rate (optional): [Dropout-rate for TDNN block. (float, default = 0.0)]

        # Transformer
        - type: transformer
          d_hidden: [Input/output dimension of Transformer block. (int)]
          d_ff: [Hidden dimension of the Feed-forward module. (int)]
          heads: [Number of heads in multi-head attention. (int)]
          dropout-rate (optional): [Dropout-rate for Transformer block. (float, default = 0.0)]
          pos-dropout-rate (optional): [Dropout-rate for positional encoding module. (float, default = 0.0)]
          att-dropout-rate (optional): [Dropout-rate for attention module. (float, default = 0.0)]

        # Conformer
        - type: conformer
          d_hidden: [Input/output dimension of Conformer block (int)]
          d_ff: [Hidden dimension of the Feed-forward module. (int)]
          heads: [Number of heads in multi-head attention. (int)]
          macaron_style: [Whether to use macaron style. (bool)]
          use_conv_mod: [Whether to use convolutional module. (bool)]
          conv_mod_kernel (required if use_conv_mod = True): [Number of kernel in convolutional module. (int)]
          dropout-rate (optional): [Dropout-rate for Transformer block. (float, default = 0.0)]
          pos-dropout-rate (optional): [Dropout-rate for positional encoding module. (float, default = 0.0)]
          att-dropout-rate (optional): [Dropout-rate for attention module. (float, default = 0.0)]

        # Causal Conv1d
        - type: causal-conv1d
          idim: [Input dimension. (int)]
          odim: [Output dimension. (int)]
          kernel_size: [Size of the context window. (int)]
          stride (optional): [Stride of the sliding blocks. (int, default = 1)]
          dilation (optional): [Parameter to control the stride of elements within the neighborhood. (int, default = 1)]
          groups (optional): [Number of blocked connections from input channels to output channels. (int, default = 1)
          bias (optional): [Whether to add a learnable bias to the output. (bool, default = True)]
          use-relu (optional): [Whether to use a ReLU activation after convolution. (bool, default = True)]
          use-batchnorm: [Whether to use batch normalization after convolution. (bool, default = False)]
          dropout-rate (optional): [Dropout-rate for TDNN block. (float, default = 0.0)]

3) The defined architecture can be repeated by specifying the total number of blocks/layers in the architecture through `enc-block-repeat` or/and `dec-block-repeat` parameters:

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

#### Multi-task learning

We also support multi-task learning with various auxiliary losses, such as: CTC, cross-entropy w/ label-smoothing (LM loss), auxiliary Transducer, and symmetric KL divergence.
The four losses can be simultaneously trained with main Transducer loss to jointly optimize the total loss defined as:

![augmented Transducer training](http://www.sciweavers.org/tex2img.php?eq=\mathcal{L}_{tot}%20%3D%20\lambda_{1}\mathcal{L}_{1}%20%2B%20\lambda_{2}\mathcal{L}_{2}%20%2B%20\lambda_{3}\mathcal{L}_{3}%20%2B%20\lambda_{4}%20\mathcal{L}_{4}%20%2B%20\lambda_{5}%20\mathcal{L}_{5}&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

where the losses are respectively, in order: The main Transducer loss, the CTC loss, the auxiliary Transducer loss, the symmetric KL divergence loss, and the LM loss. Lambda values define their respective contribution to the overall loss. Additionally, each loss can be independently selected or omitted depending on the task.

Each loss can be defined in the training config alongside its specific options, such as follow:

        # Transducer loss (L1)
        transducer-loss-weight: [Weight of the main Transducer loss (float)]

        # CTC loss (L2)
        use-ctc-loss: True
        ctc-loss-weight (optional): [Weight of the CTC loss. (float, default = 0.5)]
        ctc-loss-dropout-rate (optional): [Dropout rate for encoder output representation. (float, default = 0.0)]

        # Auxiliary Transducer loss (L3)
        use-aux-transducer-loss: True
        aux-transducer-loss-weight (optional): [Weight of the auxiliary Transducer loss. (float, default = 0.4)]
        aux-transducer-loss-enc-output-layers (required if use-aux-transducer-loss = True): [List of intermediate encoder layer IDs to compute auxiliary Transducer loss(es). (list)]
        aux-transducer-loss-mlp-dim (optional): [Hidden dimension for the MLP network. (int, default = 320)]
        aux-transducer-loss-mlp-dropout-rate: [Dropout rate for the MLP network. (float, default = 0.0)]

        # Symmetric KL divergence loss (L4)
        # Note: It can be only used in addition to the auxiliary Transducer loss.
        use-symm-kl-div-loss: True
        symm-kl-div-loss-weight (optional): [Weight of the symmetric KL divergence loss. (float, default = 0.2)]

        # LM loss (L5)
        use-lm-loss: True
        lm-loss-weight (optional): [Weight of the LM loss. (float, default = 0.2)]
        lm-loss-smoothing-rate: [Smoothing rate for LM loss. If > 0, label smoothing is enabled. (float, default = 0.0)]

#### Inference

Various decoding algorithms are also available for Transducer by setting `beam-size` and `search-type` parameters in decode config.

  - Greedy search  constrained to one emission by timestep (`beam-size: 1`).
  - Beam search algorithm without prefix search (`beam-size: >1` and `search-type: default`).
  - Time Synchronous Decoding [[Saon et al., 2020]](https://ieeexplore.ieee.org/abstract/document/9053040) (`beam-size: >1` and `search-type: tsd`).
  - Alignment-Length Synchronous Decoding [[Saon et al., 2020]](https://ieeexplore.ieee.org/abstract/document/9053040) (`beam-size: >1` and `search-type: alsd`).
  - N-step Constrained beam search modified from [[Kim et al., 2020]](https://arxiv.org/abs/2002.03577) (`beam-size: >1` and `search-type: default`).
  - modified Adaptive Expansion Search, based on [[Kim et al., 2021]](https://ieeexplore.ieee.org/abstract/document/9250505) and NSC (`beam-size: >1` and `search-type: maes`).

The algorithms share two parameters to control beam size (`beam-size`) and final hypotheses normalization (`score-norm-transducer`). The specific parameters for each algorithm are:

        # Default beam search
        search-type: default

        # Time-synchronous decoding
        search-type: tsd
        max-sym-exp: [Number of maximum symbol expansions at each time step (int)]

        # Alignement-length decoding
        search-type: alsd
        u-max: [Maximum output sequence length (int)]

        # N-step Constrained beam search
        search-type: nsc
        nstep: [Number of maximum expansion steps at each time step (int)]
               # nstep = max-sym-exp + 1 (blank)
        prefix-alpha: [Maximum prefix length in prefix search (int)]

        # modified Adaptive Expansion Search
        search-type: maes
        nstep: [Number of maximum expansion steps at each time step (int, > 1)]
        prefix-alpha: [Maximum prefix length in prefix search (int)]
        expansion-gamma: [Number of additional candidates in expanded hypotheses selection (int)]
        expansion-beta: [Allowed logp difference for prune-by-value method (float, > 0)]

Except for the default algorithm, the described parameters are used to control the performance and decoding speed. The optimal values for each parameter are task-dependent; a high value will typically increase decoding time to focus on performance while a low value will improve decoding time at the expense of performance.

#### Additional notes

- Similarly to training with CTC, Transducer does not output the validation accuracy. Thus, the optimum model is selected with its loss value (i.e., --recog_model model.loss.best).
- There are several differences between MTL and Transducer training/decoding options. The users should refer to `espnet/espnet/nets/pytorch_backend/e2e_asr_transducer.py` for an overview and `espnet/espnet/nets/pytorch_backend/transducer/arguments` for all possible arguments.
- FastEmit regularization [[Yu et al., 2021]](https://arxiv.org/pdf/2010.11148) is available through `--fastemit-lambda` training parameter (default = 0.0).
- RNN-decoder pre-initialization using an LM is supported. Note that regular decoder keys are expected. The LM state dict keys (`predictor.*`) will be renamed according to AM state dict keys (`dec.*`).
- Transformer-decoder pre-initialization using a Transformer LM is not supported yet.

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

ESPnet currently supports two finetuning operations: transfer learning and freezing.
We expect the user to define the following options in its main training config (e.g.: conf/train*.yaml). If needed, they can be directly passed to `(asr|tts|vc)_train.py` by adding the prefix `--` to the options.

#### Transfer learning

- Transfer learning option is split between encoder initialization (`enc-init`) and decoder initialization (`dec-init`). However, the same model can be specified for both options.
- Each option takes a snapshot path (e.g.: `[espnet_model_path]/results/snapshot.ep.1`) or model path (e.g.: `[espnet_model_path]/results/model.loss.best`) as argument.
- Additionally, a list of encoder and decoder modules (separated by a comma) can also be specified to control the modules to transfer with the options `enc-init-mods` and `dec-init-mods`.
- For each specified module, we only expect a partial match with the start of the target model module name. Thus, multiple modules can be specified with the same key if they share a common prefix.

    > Mandatory: `enc-init: /home/usr/espnet/egs/vivos/asr1/exp/train_nodev_pytorch_train/results/model.loss.best` -> specify a pre-trained model on VIVOS for transfer learning.  
         > Example 1: `enc-init-mods: 'enc.'` -> transfer all encoder parameters.  
         > Example 2: `enc-init-mods: 'enc.embed.,enc.0.'` -> transfer encoder embedding layer and first layer parameters.  


#### Freezing

- Freezing option can be enabled with `freeze-mods`, (`freeze_param` in espnet2).
- The option take a list of model modules (separated by a comma) as argument. As previously, we do not expect a complete match for the specified modules.

    > Example 1: `freeze-mods: 'enc.embed.'` -> freeze encoder embedding layer parameters.  
    > Example 2: `freeze-mods: 'dec.embed,dec.0.'` -> freeze decoder embedding layer and first layer parameters.
    > Example 3 (espnet2): `freeze_param: 'encoder.embed'` -> freeze encoder embedding layer parameters.

### Important notes

- Given a pre-trained source model, the modules specified for transfer learning are expected to have the same parameters (i.e.: layers and units) as the target model modules.
- We also support initialization with a pre-trained RNN LM for the RNN-Transducer decoder.
- RNN models use different key names for encoder and decoder parts compared to Transformer, Conformer or Custom models:
  - RNN model use `enc.` for encoder part and `dec.` for decoder part.
  - Transformer/Conformer/Custom model use `encoder.` for encoder part and `decoder.` for decoder part.

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
