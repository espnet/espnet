# LibriSpeech 100h ASR recipe

Place the corpus under `download/LibriSpeech`, or set the `LIBRISPEECH`
environment variable to an existing LibriSpeech root, before running
`create_dataset`/`train`.

## Quick start

```bash
# 1) Train the default E-Branchformer model
python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

# 2) Decode
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml

# 3) Score
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```

## Conformer-CTC

`conf/tuning/training_conformer_ctc.yaml` ports
`egs2/librispeech_100/asr1/conf/tuning/train_conformer_ctc.yaml`: an 18-block
Conformer encoder trained with the CTC loss only (no attention decoder).

```bash
python run.py --stages create_dataset train_tokenizer collect_stats \
    --training_config conf/tuning/training_conformer_ctc.yaml

python run.py --stages train \
    --training_config conf/tuning/training_conformer_ctc.yaml

python run.py --stages infer measure \
    --training_config conf/tuning/training_conformer_ctc.yaml \
    --inference_config conf/inference_ctc.yaml \
    --metrics_config conf/metrics.yaml
```

Run `collect_stats` and `train` as separate commands: `collect_stats` removes
`model.normalize` from the training config it was given, so a `train` stage
in the same command builds the model without global mean/variance
normalization.

It uses the same data, BPE-5000 tokenizer and frontend as
`training_e_branchformer.yaml`, so `data/` and `exp/stats` are shared: if the
other config has already run, `train_tokenizer` and `collect_stats` can be
skipped.

Without a decoder `valid/acc` is not logged, so checkpoints are ranked by
`valid/cer_ctc` and the 10 best are averaged into
`exp/training_conformer_ctc/valid.cer_ctc.ave_10best.pth`.
`conf/inference_ctc.yaml` decodes with the CTC prefix scorer alone
(`ctc_weight: 1.0`, `beam_size: 1`, as egs2's `decode_ctc_bs1.yaml`) and writes
to `exp/training_conformer_ctc/inference_ctc/<test set>/`.

Differences from the egs2 config (each is marked `[DEVIATION]` in the config):

- `frontend_conf` uses `n_fft` 512 / `win_length` 400 / `hop_length` 160
  instead of the espnet2 default (`win_length` 512, `hop_length` 128), so that
  `exp/stats` is shared with `training_e_branchformer.yaml`.
- egs2 batches with `batch_bins: 2000000` raw samples and `accum_grad: 16`.
  ESPnet3 bins on `feats_shape` (frames x 80), so this config uses
  `batch_bins: 4000000` (about 500 s of audio) with
  `accumulate_grad_batches: 4`, keeping about 2,000 s of audio per update.
- Speed perturbation (0.9/1.0/1.1) is applied on the fly by the preprocessor
  instead of tripling the training set on disk. An epoch therefore covers
  100 h instead of 300 h, so `max_epochs` is 180 instead of 60 to keep the
  same number of updates.

### Results

| dataset | WER | CER |
| --- | --- | --- |
| dev-clean | 10.45 | 4.43 |
| dev-other | 23.95 | 12.49 |
| test-clean | 10.72 | 4.53 |
| test-other | 25.18 | 12.94 |

Trained for all 180 configured epochs on 1 NVIDIA A100 (40 GB) in bf16, at
about 3.9 min per epoch (31.7 M parameters). Training was interrupted once
after epoch 30 and resumed from the step checkpoint with `fit.ckpt_path=last`;
the step-based `WarmupLR` schedule continues unchanged across the restart.
Decoded with `conf/inference_ctc.yaml` (CTC only, `beam_size` 1, no LM) from
the `valid.cer_ctc.ave_10best` average of epochs 156-178 (0-based, as in the
checkpoint file names).

egs2 reports no result for `train_conformer_ctc.yaml`. Its closest CTC-only
baseline, `train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml` (26.4 M
parameters, `decode_ctc_bs1`), reaches WER 9.2 / 22.4 / 9.6 / 23.1 on the same
four sets.

## Pretrained Models

- [`conf/tuning/training_e_branchformer.yaml`](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
- [`conf/tuning/training_conformer_ctc.yaml`](https://huggingface.co/YuCeong-May/Conformer-CTC)
