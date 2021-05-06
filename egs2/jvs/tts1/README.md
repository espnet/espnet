# JVS RECIPE

This is the recipe of the adaptation with Japanese single speaker in [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) corpus.

This recipe assumes the use of pretrained model.
Please follow the usage to perform fine-tuning with pretrained model.

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

## How to run

Here, we show the procedure of the fine-tuning using Tacotron2 pretrained with [JSUT](../../jsut/tts1) corpus.

### 1. Run the recipe until stage 5

```sh
# From data preparation to statistics calculation
$ ./run.sh --stop-stage 5
```

The detail of stage 1-5 can be found in [`Recipe flow`](../../TEMPLATE/tts1/README.md#recipe-flow).

### 2. Download pretrained model

Download pretrained model from ESPnet model zoo here.
If you have your own pretrained model, you can skip this step.

```sh
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cachedir downloads kan-bayashi/jsut_tacotron2
```

You can find the other pretrained models in [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv).

### 3. Replace token list and statistics with pretrained model's one

Since we use the same language data for fine-tuning, we need to use the token list of the pretrained model instead of that of data for fine-tuning.
The downloaded pretrained model has `tokens_list` in the config, so first we create `tokens.txt` (`token_list`) from the config.

```sh
# NOTE: The path may be changed. Please change it to match with your case.
$ pyscripts/utils/make_token_list_from_config.py downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk/config.yaml

# tokens.txt is created in model directory
$ ls downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk
199epoch.pth    config.yaml   tokens.txt
```

Let us replace the `tokens.txt` and `feats_stats.npz` with pretrained model's one.
```sh
# NOTE: The path may be changed. Please change it to match with your case.

# Make backup (Rename -> *.bak)
$ mv data/token_list/phn_jaconv_pyopenjtalk/tokens.{txt,txt.bak}
$ mv exp/tts_stats_raw_phn_jaconv_pyopenjtalk/train/feats_stats.{npz,npz.bak}

# Make symlink to pretrained model's one (Just copy is also OK)
$ ln -s downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk/tokens.txt data/token_list/phn_jaconv_pyopenjtalk
$ ln -s downloads/2dc62478870c846065fe39e609ba6657/exp/tts_stats_raw_phn_jaconv_pyopenjtalk/train/feats_stats.npz exp/tts_stats_raw_phn_jaconv_pyopenjtalk/train
```

Now ready to perform fine-tuning!

### 4. Run fine-tuning

Run the recipe from stage 6.

You need to specify `--init_param` for `--train_args` to load pretrained parameters (Or you can write them in `*.yaml` config).

```sh
# NOTE: The path may be changed. Please change it to match with your case.

# Recommend using --tag to name the experiment directory
$ ./run.sh --stage 6 --train_config conf/tuning/finetune_tacotron2.yaml \
    --train_args "--init_param downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk/199epoch.pth" \
    --tag finetune_jsut_pretrained_tacotron2
```

If you want to load part of the pretrained model, please see [`How to load pretrained model?`](../../TEMPLATE/tts1/README.md#how-to-load-the-pretrained-model) For example, if you want to perform fine-tuning of English model with Japanese data, you may want to load the network except for the token embedding layer.
