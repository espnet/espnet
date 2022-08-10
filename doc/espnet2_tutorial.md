# ESPnet2
We are planning a super major update, called `ESPnet2`. The developing status is still **under construction** yet, so please be very careful to use with understanding following cautions:

- There might be fatal bugs related to essential parts.
- We haven't achieved comparable results to espnet1 on each task yet.

## Main changing from ESPnet1

- **Chainer free**
  - Discarding Chainer completely.
  - The development of Chainer is stopped at v7: https://chainer.org/announcement/2019/12/05/released-v7.html
- **Kaldi free**
   - It's not mandatory to compile Kaldi.
   - **If you find some recipes requiring Kaldi mandatory, please report it. It should be dealt with as a bug in ESPnet2.**
   - We still support the features made by Kaldi optionally.
   - We still follow Kaldi style. i.e. depending on `utils/` of Kaldi.
- **On the fly** feature extraction & text preprocessing for training
   - You don't need to create the feature file before training, but just input wave data directly.
   - We support both raw wave input and extracted features.
   - The preprocessing for text, tokenization to characters, or sentencepieces, can be also applied during training.
   - Support **self-supervised learning representations** from s3prl
- Discarding the JSON format describing the training corpus.
   - Why do we discard the JSON format? Because a dict object generated from a large JSON file requires much memory and it also takes much time to parse such a large JSON file.
- Support distributed data-parallel training (Not enough tested)
   - Single node multi GPU training with `DistributedDataParallel` is also supported.

## Recipes using ESPnet2

You can find the new recipes in `egs2`:

```
espnet/  # Python modules of epsnet1
espnet2/ # Python modules of epsnet2
egs/     # espnet1 recipes
egs2/    # espnet2 recipes
```

The usage of recipes is **almost the same** as that of ESPnet1.


1. Change directory to the base directory

    ```bash
    # e.g.
    cd egs2/an4/asr1/
    ```
    `an4` is a tiny corpus and can be freely obtained, so it might be suitable for this tutorial. 
    You can perform any other recipes as the same way. e.g. `wsj`, `librispeech`, and etc.

    Keep in mind that all scripts should be ran at the level of `egs2/*/{asr1,tts1,...}`.
    
    ```bash
    # Doesn't work
    cd egs2/an4/
    ./asr1/run.sh
    ./asr1/scripts/<some-script>.sh
    
    # Doesn't work
    cd egs2/an4/asr1/local/
    ./data.sh
    
    # Work
    cd egs2/an4/asr1
    ./run.sh
    ./scripts/<some-script>.sh
    ```
    
1. Change the configuration
    Describing the directory structure as follows:
    
    ```
    egs2/an4/asr1/
     - conf/      # Configuration files for training, inference, etc.
     - scripts/   # Bash utilities of espnet2
     - pyscripts/ # Python utilities of espnet2
     - steps/     # From Kaldi utilities
     - utils/     # From Kaldi utilities
     - db.sh      # The directory path of each corpora
     - path.sh    # Setup script for environment variables
     - cmd.sh     # Configuration for your backend of job scheduler
     - run.sh     # Entry point
     - asr.sh     # Invoked by run.sh
    ```

    - You need to modify `db.sh` for specifying your corpus before executing `run.sh`. For example, when you touch the recipe of `egs2/wsj`, you need to change the paths of `WSJ0` and `WSJ1` in `db.sh`.
    - Some corpora can be freely obtained from the WEB and they are written as "downloads/" at the initial state. You can also change them to your corpus path if it's already downloaded.
    - `path.sh` is used to set up the environment for `run.sh`. Note that the Python interpreter used for ESPnet is not the current Python of your terminal, but it's the Python which was installed at `tools/`. Thus you need to source `path.sh` to use this Python.
        ```bash
        . path.sh
        python
        ```
    - `cmd.sh` is used for specifying the backend of the job scheduler. If you don't have such a system in your local machine environment, you don't need to change anything about this file. See [Using Job scheduling system](./parallelization.md)

1. Run `run.sh`

    ```bash
    ./run.sh
    ```

    `run.sh` is an example script, which we often call as "recipe", to run all stages related to DNN experiments; data-preparation, training, and evaluation.

## See training status

### Show the log file

```bash
% tail -f exp/*_train_*/train.log
[host] 2020-04-05 16:34:54,278 (trainer:192) INFO: 2/40epoch started. Estimated time to finish: 7 minutes and 58.63 seconds
[host] 2020-04-05 16:34:56,315 (trainer:453) INFO: 2epoch:train:1-10batch: iter_time=0.006, forward_time=0.076, loss=50.873, los
s_att=35.801, loss_ctc=65.945, acc=0.471, backward_time=0.072, optim_step_time=0.006, lr_0=1.000, train_time=0.203
[host] 2020-04-05 16:34:58,046 (trainer:453) INFO: 2epoch:train:11-20batch: iter_time=4.280e-05, forward_time=0.068, loss=44.369
, loss_att=28.776, loss_ctc=59.962, acc=0.506, backward_time=0.055, optim_step_time=0.006, lr_0=1.000, train_time=0.173
```

### Show the training status in a image file

```bash
# Accuracy plot
# (eog is Eye of GNOME Image Viewer)
eog exp/*_train_*/images/acc.img
# Attention plot
eog exp/*_train_*/att_ws/<sample-id>/<param-name>.img
```

### Use tensorboard

```sh
tensorboard --logdir exp/*_train_*/tensorboard/
```

# Instruction for run.sh
## How to parse command-line arguments in shell scripts?

All shell scripts in espnet/espnet2 depend on [utils/parse_options.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh) to parase command line arguments.

e.g. If the script has `ngpu` option

```sh
#!/usr/bin/env bash
# run.sh
ngpu=1
. utils/parse_options.sh
echo ${ngpu}
```

Then you can change the value as follows:

```sh
$ ./run.sh --ngpu 2
echo 2
```

You can also show the help message:

```sh
./run.sh --help
```

## Start from a specified stage and stop at a specified stage
The procedures in `run.sh` can be divided into some stages, e.g. data preparation, training, and evaluation. You can specify the starting stage and the stopping stage.

```sh
./run.sh --stage 2 --stop-stage 6
```

There are also some altenative otpions to skip specified stages:

```sh
run.sh --skip_data_prep true  # Skip data preparation stages.
run.sh --skip_train true      # Skip training stages.
run.sh --skip_eval true       # Skip decoding and evaluation stages.
run.sh --skip_upload false    # Enable packing and uploading stages.
```

Note that `skip_upload` is true by default. Please change it to false when uploading your model.

## Change the configuration for training
Please keep in mind that `run.sh` is a wrapper script of several tools including DNN training command. 
You need to do one of the following two ways to change the training configuration. 

```sh
# Give a configuration file
./run.sh --asr_config conf/train_asr.yaml
# Give arguments to "espnet2/bin/asr_train.py" directly
./run.sh --asr_args "--foo arg --bar arg2"
```

e.g. To change learning rate for the LM training

```sh
./run.sh --lm_args "--optim_conf lr=0.1"
```

This is the case of ASR training and you need to replace it accordingly for the other task. e.g. For TTS

```sh
./run.sh --tts_args "--optim_conf lr=0.1"
```

See [Change the configuration for training](./espnet2_training_option.md) for more detail about the usage of training tools.


## Change the number of parallel jobs

```sh
./run.sh --nj 10             # Chnage the number of parallels for data preparation stages.
./run.sh --inference_nj 10   # Chnage the number of parallels for inference jobs.
```

We also support submitting jobs to multiple hosts to accelerate your experiment: See [Using Job scheduling system](./parallelization.md)


## Multi GPUs training and distributed training

```sh
./run.sh --ngpu 4 # 4GPUs in a single node
./run.sh --ngpu 2 --num_nodes 2 # 2GPUs x 2nodes
```


Note that you need to setup your environment correctly to use distributed training. See the following two:

- [Distributed training](./espnet2_distributed.md)
- [Using Job scheduling system](./parallelization.md)


### Relationship between mini-batch size and number of GPUs

The behavior of batch size in ESPnet2 during multi-GPU training is different from that in ESPnet1. **In ESPnet2, the total batch size is not changed regardless of the number of GPUs.** Therefore, you need to manually increase the batch size if you increase the number of GPUs. Please refer to this [doc](https://espnet.github.io/espnet/espnet2_training_option.html#the-relation-between-mini-batch-size-and-number-of-gpus) for more information.


## Use specified experiment directory for evaluation

If you already have trained a model, you may wonder how to give it to run.sh when you'll evaluate it later.
By default the directory name is determined according to given options, `asr_args`, `lm_args`, or etc.
You can overwrite it by `--asr_exp` and `--lm_exp`.

```sh
# For ASR recipe
./run.sh --skip_data_prep true --skip_train true --asr_exp <your_asr_exp_directory> --lm_exp <your_lm_exp_directory>

# For TTS recipe
./run.sh --skip_data_prep true --skip_train true --tts_exp <your_tts_exp_directory>
```

## Evaluation without training using pretrained model


```sh
./run.sh --download_model <model_name> --skip_train true
```

You need to fill `model_name` by yourself. You can search for pretrained models on Hugging Face using the tag [espnet](https://huggingface.co/models?library=espnet)

(Deprecated: See the following link about our pretrain models: https://github.com/espnet/espnet_model_zoo)

## Packing and sharing your trained model

ESPnet encourages you to share your results using platforms like [Hugging Face](https://huggingface.co/) or [Zenodo](https://zenodo.org/) (This last will become deprecated.)

For sharing your models, the last three stages of each task simplify this process. The model is packed into a zip file and uploaded to the selected platform (one or both).

For **Hugging Face**, you need to first create a repository (`<my_repo> = <user_name>/<repo_name>`).  
Remember to install `git-lfs ` before continuing.
Then, execute `run.sh` as follows:

```sh
# For ASR recipe
./run.sh --stage 14 --skip-upload-hf false --hf-repo <my_repo>

# For TTS recipe
./run.sh --stage 8 --skip-upload-hf false --hf-repo <my_repo>
```

For **Zenodo**, you need to register your account first. Then, execute `run.sh` as follows:

```sh
# For ASR recipe
./run.sh --stage 14 --skip-upload false

# For TTS recipe
./run.sh --stage 8 --skip-upload false
```

The packed model can be uploaded to both platforms by setting the previously mentioned flags.

## Usage of Self-Supervised Learning Representations as feature

ESPnet supports self-supervised learning representations (SSLR) to replace traditional spectrum features. In some cases, SSLRs can boost the performance.

To use SSLRs in your task, you need to make several modifications.

### Prerequisite
1. Install [S3PRL](https://github.com/s3prl/s3prl) by `tools/installers/install_s3prl.sh`.
2. If HuBERT / Wav2Vec is needed, [fairseq](https://github.com/pytorch/fairseq) should be installed by `tools/installers/install_fairseq.sh`.

### Usage
1. To reduce the time used in `collect_stats` step, please specify `--feats_normalize uttmvn` in `run.sh` and pass it as arguments to `asr.sh` or other task-specific scripts. (Recommended)
2. In the configuration file, specify the `frontend` and `preencoder`. Taking `HuBERT` as an example:
   The `upstream` name can be whatever supported in S3PRL. `multilayer-feature=True` means the final representation is a weighted-sum of all layers' hidden states from SSLR model.
   ```
   frontend: s3prl
   frontend_conf:
      frontend_conf:
         upstream: hubert_large_ll60k  # Note: If the upstream is changed, please change the input_size in the preencoder.
      download_dir: ./hub
      multilayer_feature: True
   ```
   Here the `preencoder` is to reduce the input dimension to the encoder, to reduce the memory cost. The `input_size` depends on the upstream model, while the `output_size` can be set to any values.
   ```
   preencoder: linear
   preencoder_conf:
      input_size: 1024  # Note: If the upstream is changed, please change this value accordingly.
      output_size: 80
   ```
3. Because the shift sizes of different `upstream` models are different, e.g. `HuBERT` and `Wav2Vec2.0` have `20ms` frameshift. Sometimes, the downsampling rate (`input_layer`) in the `encoder` configuration need to be changed. For example, using `input_layer: conv2d2` will results in a total frameshift of `40ms`, which is enough for some tasks.

## Streaming ASR
ESPnet supports streaming Transformer/Conformer ASR with blockwise synchronous beam search.

For more details, please refer to the [paper](https://arxiv.org/pdf/2006.14941.pdf).

### Training

To achieve streaming ASR, please employ blockwise Transformer/Conformer encoder in the configuration file. Taking `blockwise Transformer` as an example:
The `encoder` name can be `contextual_block_transformer` or `contextual_block_conformer`. 

```sh
encoder: contextual_block_transformer
encoder_conf:
    block_size: 40         # block size for block processing
    hop_size: 16           # hop size for block processing
    look_ahead: 16         # look-ahead size for block processing
    init_average: true     # whether to use average input as initial context 
    ctx_pos_enc: true      # whether to use positional encoding for the context vectors 
```
   
### Decoding

To enable online decoding, the argument `--use_streaming true` should be added to `run.sh`.

```sh
./run.sh --stage 12 --use_streaming true
```

### FAQ
1. Issue about `'NoneType' object has no attribute 'max'` during training: Please make sure you employ `forward_train` function during traininig, check more details [here](https://github.com/espnet/espnet/issues/3803).
3. I successfully trained the model, but encountered the above issue during decoding: You may forget to specify `--use_streaming true` to select streaming inference.

## Real-Time-Factor and Latency

In order to calculate real-time-factor and (non-streaming) latency the script `utils/calculate_rtf.py` has been reworked and can now be used for both ESPnet1 and ESPnet2. The script calculates inference times based on time markers in the decoding log files and reports the average real-time-factor (RTF) and average latency over all decoded utterances. For ESPnet2, the script will automatically be run (see [Limitations](#limitations) section below) after the decoding stage has finished but can also be run as a stand-alone script:

### Usage

```
usage: calculate_rtf.py [-h] [--log-dir LOG_DIR]
                        [--log-name {decode,asr_inference}]
                        [--input-shift INPUT_SHIFT]
                        [--start-times-marker {input lengths,speech length}]
                        [--end-times-marker {prediction,best hypo}]

calculate real time factor (RTF)

optional arguments:
  -h, --help            show this help message and exit
  --log-dir LOG_DIR     path to logging directory
  --log-name {decode,asr_inference}
                        name of logfile, e.g., 'decode' (espnet1) and
                        'asr_inference' (espnet2)
  --input-shift INPUT_SHIFT
                        shift of inputs in milliseconds
  --start-times-marker {input lengths,speech length}
                        String marking start of decoding in logfile, e.g.,
                        'input lengths' (espnet1) and 'speech length'
                        (espnet2)
  --end-times-marker {prediction,best hypo}
                        String marking end of decoding in logfile, e.g.,
                        'prediction' (espnet1) and 'best hypo' (espnet2)
```

### Notes

- Default settings still target ESPnet1 usage:
  ```
  --log-name 'decode'
  --input-shift 10.0
  --start-times-marker 'input lengths'
  --end-times-marker 'prediction'
  ```
- For ESPnet2, other frame shifts than 10ms are possible via different front-end/feature configurations. So different to ESPnet1, which logs the input feature frames at a fixed 10ms frame shift, in ESPnet2 the number of speech samples is logged instead and the audio sample shift in milliseconds (1/sampleRate x 1000) needs to be specified for `--input-shift` parameter (see `--input-shift 0.0625` in example below for 16000 Hz sample rate).

### Example

From ```espnet/egs2/librispeech/asr1``` the following call runs the decoding stage with pretrained ESPnet2 model:

```sh
./run.sh --stage 12  --use_streaming false --skip_data_prep true --skip_train true --download_model byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp
```
Results for latency and rtf calculation on Librispeech test_clean subset can then be found in ```espnet/egs2/librispeech/asr1/exp/byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp/decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean/logdir/calculate_rtf.log``` file:
```sh
# ../../../utils/calculate_rtf.py --log-dir exp/byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp/decode_as
r_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean/logdir --log-name asr_inference --input-shift 0.0625 --start-times-
marker "speech length" --end-times-marker "best hypo"
Total audio duration: 19452.481 [sec]
Total decoding time: 137762.231 [sec]
RTF: 7.082
Latency: 52581.004 [ms/sentence]
```

### Limitations

- Only non-streaming inference mode is supported currently
- The decoding stage 12 in `asr.sh` automatically runs the rtf & latency calculation if `"asr_inference_tool == "espnet2.bin.asr_inference"`; other inference tools like k2 & maskctc are still left to do
