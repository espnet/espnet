# ESPnet2 JVS TTS recipe

This recipe assumes that the use of pretrained model.
Please follow the usage to perform finetuning with pretrained model.

## How to run

Here, we show the procedure of the finetuing using Tacotron2 pretrained with JSUT corpus.

1. Run the recipe until stage 5

```sh
# From data preparation to statistics calculation
$ ./run.sh --stop-stage 5
```

2. Download pretrained model

Download pretrained model from ESPnet model zoo here.
If you have your own pretrained model, you can skip this step.

```sh
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cache_dir downloads kan-bayashi/jsut_tacotron2
```

3. Replace token list and statistics with pretrained model's one

Since we use the same language model, we need to use the token list of the pretrained model.
The downloaded pretrained model has no `tokens.txt` file, so first we create it from the config.

```sh
# NOTE: The path may be changed. Please change it to match with your case.
$ pyscripts/utils/create_token_list_from_config.py downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk/config.yaml
```

Then, `tokens.txt` is created.

```sh
# NOTE: The path may be changed. Please change it to match with your case.
$ ls downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk
199epoch.pth    config.yaml   tokens.txt
```

Let us replace the `tokens.txt` and `feats_stats.npz` with pretrained model's one.
```sh
# NOTE: The path may be changed. Please change it to match with your case.

# Make backup
$ mv data/token_list/phn_jaconv_pyopenjtalk/tokens.{txt,txt.bak}
$ mv exp/tts_stats_raw_phn_jaconv_pyopenjtalk/train/feats_stats.{npz,npz.bak}

# Make symlink to pretrained model's one
$ ln -s downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk/tokens.txt data/token_list/phn_jaconv_pyopenjtalk
$ ln -s downloads/2dc62478870c846065fe39e609ba6657/exp/tts_stats_raw_phn_jaconv_pyopenjtalk/train/feats_stats.npz exp/tts_stats_raw_phn_jaconv_pyopenjtalk/train
```

4. Run fine-tuning

Now ready to fine-tune. Run the recipe from stage 6.
You need to specify `--pretrain_path` and `--pretrain_key` for `--train_args` to load pretrained parameters (Or you can write them in `*.yaml` config).
If you want to load the entire network, please specify `--pretrain_key null`.

```sh
# NOTE: The path may be changed. Please change it to match with your case.

# I recommend using --tag to name the experiment directory
$ ./run.sh --stage 6 --train_config conf/tuning/finetune_tacotron2.yaml \
    --train_args "--pretrain_path downloads/2dc62478870c846065fe39e609ba6657/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk/199epoch.pth --pretrain_key null" \
    --tag finetune_jsut_pretrained_tacotron2
```

If you want to load part of the pretrained model, please see [How to load pretrained model?](../../TEMPLATE/README.md) (For example, if you want to finetune English model with Japanese data, you may want to load the network except for the token embedding layer).
