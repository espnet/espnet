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
| dev_4k | 0.93 | 2.25 |

Trained for all 35 configured epochs (mostly on 4 GPUs; a brief stretch during
epochs 32-33 ran on 2 GPUs with `accumulate_grad_batches` doubled to keep the
effective batch constant). Decoded on 1 GPU with `conf/inference.yaml` (beam
20, `ctc_weight` 0.3, `batch_size` 4) from the `valid.acc.ave_10best` average.
Decoding the same checkpoint without batching (`batch_size: null`) gives WER
2.26 / CER 0.93: `Speech2Text.batch_decode`'s padded batching is not perfectly
invariant for `Conv2d` subsampling and the legacy relative-position attention
this config uses, so a handful of hypotheses differ between the two paths --
not a decoding bug, and not a difference in the model.

Batching also makes `dev_4k` (1 GPU, single process) **2.8x faster to decode**:
31.3 min (`batch_size` 4) vs. 88.75 min (`batch_size: null`), same checkpoint,
same hardware.

`val` (39,341 utterances) -- the set egs2 also reports -- has not been
decoded.

## Pretrained Models

- [`espnet/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256`](https://huggingface.co/espnet/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256)

## Differences from the egs2 recipe

These are deliberate; each is marked `[DEVIATION]` in the training config.

- **Normalized transcripts, not `_unnorm`.** `egs2/.../run.sh` sets `norm=""`
  and then overrides it with `norm="_unnorm"`, so the effective egs2 sets are
  `train_nodev_unnorm` / `dev_4k_unnorm` / `{dev_4k_unnorm, val_unnorm}`.
  This recipe runs egs2's `norm=""` branch. `create_dataset` builds all eight
  splits, so switching only requires appending `_unnorm` to the split names in
  the training config.
- **No language model.** ESPnet3 has no LM stage, whereas egs2 trains one and
  shallow-fuses it at `lm_weight: 0.3`. The WER above is therefore LM-free and
  is not directly comparable to egs2's published number.
- **`batch_bins` is not egs2's `35000000`.** The unit differs: egs2 bins on raw
  samples as a global budget, ESPnet3 bins on `feats_shape` (frames x 80) per
  GPU. The derivation is written out in the training config.
- **`num_workers: 4`** rather than espnet2's default of 1; a batch holds several
  hundred separate file reads, and one worker starves the GPUs.
- **`model_conf.sym_space` is set to the SentencePiece word-boundary marker**
  (U+2581), which egs2 leaves at espnet2's `<space>` default. `ErrorCalculator`
  rebuilds word boundaries by replacing `sym_space` before splitting on
  whitespace; with a BPE/unigram vocabulary `<space>` never appears, so the
  replacement is a no-op and the training-time `valid/wer` degenerates into a
  sentence error rate (one wrong token scores the same as an entirely wrong
  hypothesis). This affects only the progress metrics logged during training,
  not the WER/CER reported by the `measure` stage above, which are scored on
  detokenized text.
