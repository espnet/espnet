# MultiMed-ST S2T Recipe

This recipe performs English-to-German speech translation on the [MultiMed-ST](https://huggingface.co/datasets/leduckhai/MultiMed-ST) dataset using ESPnet2 S2T and OWSM v4 small fine-tuning.

## Task

* Dataset: MultiMed-ST
* Source language: English
* Target language: German
* Task: speech translation (`st`)
* Pretrained model: `espnet/owsm_v4_small_370M`

## Reference

This recipe uses the MultiMed-ST dataset introduced in [MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation](https://arxiv.org/abs/2504.03546).

## Data preparation

Data preparation is implemented in `local/data.sh` and `local/prepare_multimed_st.py`. The script streams MultiMed-ST from Hugging Face, writes mono wav files, and creates Kaldi-style `data/train`, `data/valid`, and `data/test` directories.

For OWSM fine-tuning, each split contains `text`, `text.prev`, and `text.ctc`:

```text
text      utt_id <eng><st_deu><notimestamps> German translation
text.prev utt_id <na>
text.ctc  utt_id English source transcript
```

MultiMed-ST audio can be stereo, so the preparation script downmixes it to mono before writing wav files. Since reliable speaker turns are not provided, each utterance is treated as its own speaker by setting `utt2spk` to `utt_id utt_id`.

## OWSM v4 small assets

OWSM fine-tuning uses the pretrained tokenizer, token list, checkpoint, and feature statistics from `espnet/owsm_v4_small_370M`. These assets are prepared by `local/prepare_owsm_v4_assets.py`, which creates local symlinks and a fine-tuning config:

```text
downloads/owsm_v4_small_370M/model.pth
downloads/owsm_v4_small_370M/config.yaml
downloads/owsm_v4_small_370M/bpe.model
downloads/owsm_v4_small_370M/feats_stats.npz
data/de_token_list/bpe_unigram50000/bpe.model
data/de_token_list/bpe_unigram50000/tokens.txt
conf/finetune_owsm_v4_small.yaml
```

These files are generated automatically and are not intended to be committed.

## Stages 5-9

For OWSM fine-tuning, Stage 5 is intentionally skipped.

Stage 5 normally trains a new BPE tokenizer from the training text. This is not appropriate for OWSM fine-tuning because the model must use the pretrained OWSM BPE model and token list.

Stages 6-9 are also skipped because this recipe does not train or use an external neural LM or n-gram LM (`use_lm=false`). OWSM decoding is performed directly with the fine-tuned S2T model.

Use:

```bash
--stage 1 --stop_stage 4
```

for data preparation, then continue from Stage 10.

German ST decoding uses `conf/decode_owsm_st_de.yaml`:

```yaml
lang_sym: "<eng>"
task_sym: "<st_deu>"
predict_time: false
```

For English-to-German speech translation, `lang_sym` is the source speech language and `task_sym` specifies speech translation into German.

## How to run

### Data preparation

```bash
./run.sh \
  --finetune_owsm_v4_small true \
  --stage 1 \
  --stop_stage 4 \
  --ngpu 0
```

### OWSM v4 small fine-tuning

```bash
./run.sh \
  --finetune_owsm_v4_small true \
  --stage 10 \
  --stop_stage 11 \
  --ngpu 1
```

The default generated fine-tuning configuration uses a preliminary fixed-step setup:

```text
max_epoch: 1
num_iters_per_epoch: 1000
batch_size: 1
accum_grad: 8
use_amp: true
```

For a longer local training run, override the generated-config knobs from `run.sh`. For example, the following command removes the fixed iteration limit:

```bash
./run.sh \
  --finetune_owsm_v4_small true \
  --stage 10 \
  --stop_stage 11 \
  --ngpu 1 \
  --owsm_max_epoch 5 \
  --owsm_num_iters_per_epoch 0
```

### Decoding and scoring

```bash
./run.sh \
  --finetune_owsm_v4_small true \
  --stage 12 \
  --stop_stage 13 \
  --ngpu 1 \
  --gpu_inference true \
  --inference_nj 1 \
  --inference_s2t_model 5epoch.pth
```

Stage 13 reports ESPnet's WER/CER/TER outputs and additionally calls `local/score.sh` to report ST BLEU, chrF, and TER with `sacrebleu`.

## Results

The following results are intended to validate the recipe and compare OWSM v4 small zero-shot decoding with fine-tuning on MultiMed-ST, not to report a fully optimized benchmark. The fine-tuned model was initialized from `espnet/owsm_v4_small_370M`.

### ST metrics

These scores are computed by `local/score.sh` after stripping the OWSM prompt tokens from the reference.

| Model | Test set | # utts | BLEU | chrF2 | TER |
| --- | --- | ---: | ---: | ---: | ---: |
| OWSM v4 small zero-shot | test | 4751 | 29.4 | 56.3 | 60.6 |
| OWSM v4 small fine-tuned | test | 4751 | 31.4 | 58.0 | 58.0 |

### ESPnet default scoring

These are the default Stage 13 WER/CER/TER outputs from `s2t.sh` and are included for reference.

| Model | Test set | # utts | WER | CER | TER |
| --- | --- | ---: | ---: | ---: | ---: |
| OWSM v4 small zero-shot | test | 4751 | 63.1 | 49.0 | 62.9 |
| OWSM v4 small fine-tuned | test | 4751 | 60.6 | 47.8 | 61.1 |
