# Self-supervised Learning: WavLM

This is a template of the `wavlm1` recipe for ESPnet2, designed for
[WavLM](https://arxiv.org/abs/2110.13900)-style SSL pre-training.

## Relation to the other SSL templates

`wavlm1` is the `hubert1` template with the two changes that define WavLM:

1. **Encoder** — `torchaudio_wavlm`
   (`espnet2/asr/encoder/wavlm_encoder.py`) replaces `torchaudio_hubert`. It is
   the same convolutional feature extractor and Transformer stack, but the
   self-attention adds WavLM's gated relative position bias.
2. **Masked speech denoising** — each primary utterance is mixed, on the data
   path, with either another utterance from the same batch (separation) or a
   sampled acoustic noise (denoising), while the k-means targets remain those of
   the clean primary. This lives in `HuBERTCollateFn` and is switched on with
   `collate_fn_conf.mix_speech: true` in the training config.

Everything else — the iterative offline pipeline of "dump features -> learn
k-means -> write pseudo-labels -> masked prediction training" — is shared with
`hubert1`, so `wavlm.sh` takes the same options as `hubert.sh` and the stage
numbering is identical:

| Stage | Action |
| ----- | ------ |
| 1-4   | Data preparation, wav formatting, length filtering |
| 5     | K-means on MFCC (iter 0) or on WavLM layer features (iter >= 1), then pseudo-labels + token list |
| 6     | Collect stats |
| 7     | WavLM pre-training |
| 8-9   | Pack model / upload to Hugging Face |

Stages 5-7 are repeated for each iteration between `--train_start_iter` and
`--train_stop_iter`.

See `egs2/librispeech/wavlm1` for a worked example.
