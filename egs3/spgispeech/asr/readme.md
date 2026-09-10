# SPGISpeech ASR recipe

Port of [`egs2/spgispeech/asr1`](../../../egs2/spgispeech/asr1) to ESPnet3.
SPGISpeech is 5,000 h of transcribed financial-earnings-call audio
(1,966,109 utterances, 5.01–15.00 s each).

The corpus is not downloadable automatically. Point the `SPGISPEECH`
environment variable at an existing SPGISpeech root — the directory holding
`train.csv`, `val.csv` and `spgispeech/{train,val}/` — before running
`create_dataset`:

```bash
export SPGISPEECH=/path/to/spgispeech
```

`dev_4k` and `train_nodev` are not separate files: following
`egs2/spgispeech/asr1/local/data.sh`, `train.csv` is sorted by utterance id and
the first 4,000 rows become `dev_4k`, the remainder `train_nodev`
(`dataset/config.yaml: dev_size`).

## Quick start

```bash
# 1) Build the audio index (validates every file; writes data/hf/)
python run.py --stages create_dataset \
    --training_config conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml

# 2) Train the unigram-5000 tokenizer
python run.py --stages train_tokenizer \
    --training_config conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml

# 3) Collect feature statistics (global_mvn + batch shapes)
python run.py --stages collect_stats \
    --training_config conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml

# 4) Train
python run.py --stages train \
    --training_config conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml

# 5) Decode and score
python run.py --stages infer measure \
    --training_config conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`collect_stats` over 1.96 M utterances is slow with a single local worker. Set
`parallel.env` to a cluster backend and supply your scheduler's `options`
(queue, account, walltime) in the training config.

## Results

| dataset | CER | WER |
| --- | --- | --- |
| dev_4k | 1.01 | 2.42 |

Decoded with `conf/inference.yaml` (beam 20, `ctc_weight` 0.3) from the
`valid.acc.ave_10best` average after 25 of 35 epochs on 4 GPUs.

## Pretrained Models

None published yet. The evaluation above is a partial result: it covers only
`dev_4k` (4,000 utterances) from a run stopped at 25 of the configured 35
epochs, and `val` (39,341 utterances) -- the set egs2 reports -- has not been
decoded. A model will be packaged and uploaded once training completes and the
full test sets have been scored.

## Differences from the egs2 recipe

These are deliberate; each is marked `[DEVIATION]` in the training config.

- **Normalized transcripts, not `_unnorm`.** `egs2/.../run.sh` sets `norm=""`
  and then overrides it with `norm="_unnorm"`, so the effective egs2 sets are
  `train_nodev_unnorm` / `dev_4k_unnorm` / `{dev_4k_unnorm, val_unnorm}`.
  `dataset/builder.py` restricts the cache to
  `["val", "dev_4k", "train_nodev", "train"]`, so this recipe runs egs2's
  `norm=""` branch. To use the unnormalized variant, add the four `*_unnorm`
  names to `_HF_CACHE_SPLITS`, rebuild the cache, and append `_unnorm` to the
  split names in the training config.
- **No language model.** ESPnet3 has no LM stage, whereas egs2 trains one and
  shallow-fuses it at `lm_weight: 0.3`. The WER above is therefore LM-free and
  is not directly comparable to egs2's published number.
- **`batch_bins` is not egs2's `35000000`.** The unit differs: egs2 bins on raw
  samples as a global budget, ESPnet3 bins on `feats_shape` (frames x 80) per
  GPU. The derivation is written out in the training config.
- **`num_workers: 4`** rather than espnet2's default of 1; a batch holds several
  hundred separate file reads, and one worker starves the GPUs.
