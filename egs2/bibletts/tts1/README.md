# BIBLE-TTS RECIPE

This is the recipe of single speaker TTS model with [Bible TTS](https://masakhane-io.github.io/bibleTTS/) corpus.

Our goal is to build a TTS character based system using high-quality BibleTTS dataset. BibleTTS is a large high-quality open Text-to-Speech dataset with up to 80 hours of single speaker. It releases aligned speech and text for six languages spoken in Sub-Saharan Africa. There are two options:
1) Train VITS models from scratch for each language.

2) Finetune from a pretrained TTS model to accelarate training stage.

## Recipe flow

### 1. Data preparation

Data preparation stage

Donwload dataset from  [Bible TTS](https://masakhane-io.github.io/bibleTTS/), and then run:

```sh
# Assume that data prep stage (stage 1) is finished
$ ./run.sh --stage 1 --stop-stage 1
```

### 2. VITS training
If you want to train from scratch, run:
```sh
# Specify the language name for training (e.g. Ewe, Hausa, Lingala, Yoruba, Asante-Twi, Akuapem-Twi)
$ lang=Yoruba
$ cd egs2/bibletts/tts1
$ ./run.sh \
    --train_set ${lang}/tr_no_dev \
    --valid_set ${lang}/dev1 \
    --test_sets "${lang}/dev1 ${lang}/eval1" \
    --srctexts "data/${lang}/tr_no_dev/text" \
    --dumpdir dump/${lang} \
    --stage 2 \
    --ngpu 4 \
    --g2p none \
    --token_type char \
    --cleaner none \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --inference_model train.total_count.ave.pth \
    --tag vits_bibletts_${lang}

```
If you want to finetune from a pretrain model, first download a [pretrained model](https://zenodo.org/record/5555690), and then run:

```sh
# exclude the embedding layer since we are finetuning on a different language

$ lang=yoruba
$ cd egs2/bibletts/tts1
$ ./run.sh \
    --train_set ${lang}/tr_no_dev \
    --valid_set ${lang}/dev1 \
    --test_sets "${lang}/dev1 ${lang}/eval1" \
    --srctexts "data/${lang}/tr_no_dev/text" \
    --dumpdir dump/${lang} \
    --stage 2 \
    --ngpu 4 \
    --g2p none \
    --token_type char \
    --cleaner none \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --train_args "--init_param <pretrain-model-path>":::tts.generator.text_encoder.emb \
    --inference_model train.total_count.ave.pth \
    --tag vits_lj_train_phn_1Minter_char_ft_${lang}

```
### 3. Inference

TTS model decoding stage. You can change the decoding setting via --inference_config and --inference_args.
```sh
$ lang=yoruba
$ ./run.sh \
    --train_set ${lang}/tr_no_dev \
    --valid_set ${lang}/dev1 \
    --test_sets "${lang}/dev1 ${lang}/eval1" \
    --srctexts "data/${lang}/tr_no_dev/text" \
    --dumpdir dump/${lang} \
    --ngpu 4 \
    --stage 7 \
    --min_wav_duration 0.38 \
    --g2p none \
    --token_type char \
    --cleaner none \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config conf/tuning/train_vits.yaml \
    --inference_model <checkpoint-**.pth> \
    --tag vits_bibletts_${lang}
```

### 4. Objective Evaluation

```sh
# Evaluate MCD
./pyscripts/utils/evaluate_mcd.py \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    dump/raw/eval1/wav.scp

# Evaluate log-F0 RMSE
./pyscripts/utils/evaluate_f0.py \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    dump/raw/eval1/wav.scp
```
