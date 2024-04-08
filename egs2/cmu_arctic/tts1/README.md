# CMU ARCTIC RECIPE

This is the recipe of the TTS model with the [CMU ARCTIC](http://www.festvox.org/cmu_arctic/) databases.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

## Pretrain-Finetune Setup
This recipe can be used to finetune TTS model. You can either pretrain a model using all cmu-arctic voices or use external dataset trained pretrain model.

### 1. Pretrain using cmu-arctic voices

#### 1. Data Preparation
Run the script `./run_pre_fine.sh` until stage 5

```sh
# From data preparation to statistics calculation
$ ./run_pre_fine.sh --stop-stage 5
```
The detail of stage 1-5 can be found in [`Recipe flow`](../../TEMPLATE/tts1/README.md#recipe-flow).

#### 2. Train pretrain model
Run the script `./run_pre_fine.sh` with stage 6 with pretrain_stage as True

```sh
# Recommend using --tag to name the pretrain experiment directory
$ ./run_pre_fine.sh --stage 6
```
#### 3. Finetune model
Run the script ./run_pre_fine.sh with stage 6 with finetune_stage as True and pretrain_stage as False. Make sure `--init_param` for `--train_args` points to correct pretrain model path.

```sh
# Recommend using --tag to name the finetune experiment directory
$ ./run_pre_fine.sh --stage 6
```

### 2. Finetune using external pretrain model
This part is similar to [jvs recipe example](../../jvs/tts1/README.md).

#### 1. Data Preparation
Run the recipe until stage 5

```sh
# From data preparation to statistics calculation
$ ./run.sh --stop-stage 5 --g2p pyopenjtalk_accent_with_pause
```
#### 2. Download pretrained model
Download pretrained model from ESPnet model zoo here.
If you have your own pretrained model, you can skip this step.

```sh
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cachedir downloads kan-bayashi/ljspeech_transformer
```
You can find the other pretrained models in [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv).

#### 3. Replace token list with pretrained model's one
Since we use the same language data for fine-tuning, we need to use the token list of the pretrained model instead of that of data for fine-tuning. Follow [jvs recipe example](../../jvs/tts1/README.md).

#### 4. Finetune model
Run the recipe from stage 6.
You need to specify `--init_param` for `--train_args` to load pretrained parameters (Or you can write them in `*.yaml` config).
Here `--init_param /path/to/model.pth:a:b` represents loading "a" parameters in model.pth into "b", and `:tts:tts` means load parameters except for the feature normalizer.

```sh
# Recommend using --tag to name the experiment directory
$ ./run.sh \
    --stage 6 \
    --g2p pyopenjtalk_accent_with_pause \
    --train_config conf/tuning/finetune_tacotron2.yaml \
    --train_args "--init_param downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause
```
