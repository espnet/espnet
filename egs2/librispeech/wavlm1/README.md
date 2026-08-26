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

================================================

## RESULTS

Not yet available in this repository — pre-train the model and record the
k-means quality, masked-prediction accuracy and downstream WER here.

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
