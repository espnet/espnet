## INTRODUCTION

This recipe pre-trains a [WavLM](https://arxiv.org/abs/2110.13900) model on
LibriSpeech 960h, including the k-means-based pseudo-label generation and the
masked-prediction training.

It is the `librispeech/hubert1` recipe with the two changes that define WavLM:

1. **Encoder** — `torchaudio_wavlm`
   (`espnet2/asr/encoder/wavlm_encoder.py`): the same convolutional feature
   extractor and Transformer stack as `torchaudio_hubert`, plus WavLM's gated
   relative position bias in self-attention.
2. **Masked speech denoising** — each primary utterance is, with probability
   `noise_apply_prob`, mixed with a random segment of a second source while the
   k-means targets stay those of the clean primary, forcing the model to
   denoise/separate the dominant speaker. This runs in `HuBERTCollateFn` and is
   enabled with `collate_fn_conf.mix_speech: true` in the training config.

The objective itself is unchanged from HuBERT (predict the cluster id of masked
frames), so `wavlm.sh` is `hubert.sh` with the same options and stage numbering.

================================================

## HOW TO RUN

```sh
./run.sh                       # all stages, both iterations
./run.sh --stage 5             # skip data preparation
./run.sh --stage 5 --stop_stage 5 --train_stop_iter 0   # only iteration-0 k-means
```

| Stage | Action |
| ----- | ------ |
| 1-4   | Data download/preparation, wav formatting, long/short filtering |
| 5     | K-means on MFCC (iter 0) or on WavLM layer 6 (iter 1), then pseudo-labels + token list |
| 6     | Collect stats |
| 7     | WavLM pre-training |
| 8-9   | Pack model / upload to Hugging Face (skipped by default) |

Stages 5-7 run once per iteration between `--train_start_iter` and
`--train_stop_iter`.

### Reusing the HuBERT recipe's data preparation

Stages 1-4 and the iteration-0 MFCC k-means are identical to `hubert1`, so if
that recipe has already been run there is no need to redo them:

```sh
cp -a ../hubert1/dump/raw dump/
cp -a ../hubert1/data/en_token_list_kmeans_iter0_mfcc_100clusters data/
cp -a ../hubert1/exp/kmeans_iter0_mfcc_train_960_portion0.1 exp/
ln -s ../../hubert1/data/librispeech_phoneme_alignment data/librispeech_phoneme_alignment
./run.sh --stage 6 --train_stop_iter 0
```

Note that the pseudo-label file name embeds the k-means tag
(`text.km.kmeans_iter0_mfcc_train_960_portion0.1`), which does not depend on the
recipe, so the copied labels are picked up as-is.

### GPU memory

WavLM's gated relative position bias materializes a
`(batch, heads, frames, frames)` tensor in every layer, which HuBERT's attention
does not. The configs here therefore use ~2/3 of the `hubert1` `batch_bins` with
`accum_grad` raised to keep the same effective batch. `batch_bins` is the knob to
turn if you hit OOM.

Measured on 8 x 80GB: iteration 0 peaks at **58.6 GB** (over-conservative — there
is room to raise `batch_bins` toward 40-44M) while iteration 1 peaks at
**77.1 GB**, much tighter despite a smaller `batch_bins`, because
`label_downsampling: 1` doubles the effective sequence length and the
position-bias tensor grows with its square. The two iterations have genuinely
different memory profiles; tune them separately.

================================================

## RESULTS

Pre-trained on LibriSpeech 960h, 8 x 80GB GPUs, `train_960` / `dev`.
Total wall clock 4d 19h (of which ~4d 3h was training).

### K-means teacher quality

Measured against the MFA phoneme alignments by stage 5
(`exp/kmeans_iter*/phoneme_pseudo_label_quality.txt`):

| teacher | phone purity | MI / H(ref) | MI |
| --- | --- | --- | --- |
| iter 0 — MFCC, 100 clusters | 0.2875 | 0.2475 | 0.8383 |
| iter 1 — WavLM layer 6, 500 clusters | **0.5430** | **0.5501** | **1.8630** |

Phone purity and normalized mutual information both roughly double, i.e. the
iteration-0 model's own layer-6 features are a much better clustering target
than MFCCs — which is the point of the iterative refinement.

### Pre-training

| | iter 0 | iter 1 |
| --- | --- | --- |
| targets | 100 clusters | 500 clusters |
| stopped at | ep130 (patience) | ep221 (patience) |
| best epoch | ep104 | ep195 |
| valid masked acc | 0.457 | 0.414 |
| per-frame valid loss | 1.864 nats (chance 4.615) | ~1.53 nats (chance 6.217) |
| epoch wall time | ~13 min | ~19 min |
| peak GPU memory | 58.6 GB | 77.1 GB |

Masked accuracy is NOT comparable across iterations — the class count differs, so
chance moves from 1/101 to 1/501. Normalized, iteration 1 reaches ~25% of its
chance-loss ceiling versus iteration 0's ~40%.

Downstream ASR fine-tuning results are not yet available.

### Notes for reproducing

- `patience: 25` cut iteration 0 from the 250-epoch cap to 130 (~26 h saved).
  Iteration 1's tail kept producing sub-1% improvements that reset the counter,
  so it nearly ran to the cap anyway; patience cannot catch that case, only a
  lower `max_epoch` can.
- Stage 5's k-means fit took 3h55m, 14x longer than the 17-minute GPU feature
  dump that feeds it, because it is a single scikit-learn job. Consider
  `scripts/feats/feats_clustering_cuml.sh` for GPU clustering if this matters.
- The ~500 GB of dumped layer-6 features in `dump/wavlm_feats` are only needed to
  fit the k-means; they can be deleted afterwards and regenerated in ~17 min.

================================================

## ALTERNATIVE: continue pre-training a Hugging Face WavLM

`conf/tuning/train_ssl_wavlm_base_960h_pretrain.yaml` wraps
`microsoft/wavlm-base` instead of training from scratch, for continued
pre-training or domain adaptation. It requires `transformers`
(`tools/installers/install_transformers.sh`) and is *not* a reproduction of WavLM
pre-training, since the checkpoint it starts from is already the result of it.

================================================

## REFERENCES

- WavLM paper: https://arxiv.org/abs/2110.13900
- Original code and models: https://github.com/microsoft/unilm/tree/master/wavlm
- HuBERT (shared objective and pipeline): https://arxiv.org/abs/2106.07447

## ACKNOWLEDGEMENT

This recipe builds directly on the ESPnet HuBERT recipe; see
`egs2/librispeech/hubert1/README.md` for its acknowledgements. The WavLM
Transformer comes from torchaudio's `wavlm_model` components.
