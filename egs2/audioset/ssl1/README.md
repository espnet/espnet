# BEATs Pre-training on AudioSet-2M

This recipe reproduces [BEATs](https://arxiv.org/abs/2212.09058) audio self-supervised
pre-training on AudioSet-2M in ESPnet. It iteratively trains a Transformer audio encoder
together with a VQ acoustic tokenizer, following the paper's Table 4 setup. The same
pipeline underpins [OpenBEATs](https://arxiv.org/abs/2507.14129), which scales it to a
larger multi-domain corpus.

## Installation

BEATs pre-training runs on DeepSpeed with `bfloat16`. We recommend also installing
[Flash Attention](https://github.com/Dao-AILab/flash-attention).

```
. ./path.sh
DS_BUILD_FUSED_ADAM=1 pip install deepspeed
pip install flash-attn --no-build-isolation
```

`bfloat16` requires Ampere-or-newer GPUs (e.g. A100/H200). To train in `fp16` instead,
replace the `"bf16"` block in `conf/ds_beats.json` and `conf/ds_beats_tok.json` with
`"fp16": {"enabled": true}`.

## Steps to run

Set `AUDIOSET` in `db.sh` to your AudioSet root, then run the full pipeline:

```
./run.sh
```

BEATs pre-training is **iterative** (`train_start_iter` / `train_stop_iter` in `run.sh`):
iter 0 trains the encoder against random-projection targets; iter 1 trains a VQ tokenizer
from the iter-0 encoder and re-trains the encoder against its discrete tokens. The default
config trains the base model (12-layer, 768-dim) on 2×H200 with bf16 DeepSpeed
(~33 min/epoch, 56 encoder epochs ≈ the paper's 400K-step schedule).

## Pretrained models

The extracted `beats_encoder_iter*.pt` loads directly into the downstream classification
recipes (`egs2/esc50/asr1`, `egs2/as20k/cls1`) via `beats_ckpt_path`.

| Model | Link |
|---|---|
| BEATs encoder, iter 0 | [jaeyeonkim99/BEATs-base-AS2M-iter0](https://huggingface.co/jaeyeonkim99/BEATs-base-AS2M-iter0) |
| BEATs encoder, iter 1 | [jaeyeonkim99/BEATs-base-AS2M-iter1](https://huggingface.co/jaeyeonkim99/BEATs-base-AS2M-iter1) |
| BEATs tokenizer, iter 1 | [jaeyeonkim99/BEATs-tokenizer-AS2M-iter1](https://huggingface.co/jaeyeonkim99/BEATs-tokenizer-AS2M-iter1) |

# RESULTS

## Environments
- date: `Tue Jul 21 22:21:36 CDT 2026`
- python version: `3.10.20`
- espnet version: `202604`
- pytorch version: `2.6.0+cu126`
- Git hash: `df424d24c4bcd593e93b54b62584abc6b03606e5`
- Pre-training config: [conf/beats_base.yaml](conf/beats_base.yaml)
- Tokenizer config: [conf/tok_beats_base.yaml](conf/tok_beats_base.yaml)

## Downstream evaluation

Fine-tuning the pre-trained encoder on two audio classification benchmarks
(AS-20K: `egs2/as20k/cls1`; ESC-50 5-fold: `egs2/esc50/asr1`):

| Iteration | AS-20K test mAP | ESC-50 acc (5-fold) |
|---|---|---|
| iter 0 | 29.72 | 93.35 |
| iter 1 | **31.26** | 93.35 |

The second iteration improves AS-20K by ~1.5 mAP, consistent with the gains BEATs reports
from iterative tokenizer refinement. ESC-50 (400 clips / fold) is near saturation and does
not separate the iterations.

## References

```bibtex
@inproceedings{chen2022beats,
  title={BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author={Chen, Sanyuan and Wu, Yu and Wang, Chengyi and Liu, Shujie and
          Tompkins, Daniel and Chen, Zhuo and Wei, Furu},
  booktitle={ICML},
  year={2023}
}

@inproceedings{bharadwaj2025openbeats,
  title={OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder},
  author={Bharadwaj, Shikhar and Cornell, Samuele and Choi, Kwanghee and
          Fukayama, Satoru and Shim, Hye-jin and Deshmukh, Soham and Watanabe, Shinji},
  booktitle={WASPAA},
  year={2025}
}
```

## Acknowledgement

The BEATs encoder/tokenizer implementation is ported from the original
[BEATs](https://github.com/microsoft/unilm/tree/master/beats) release. This recipe was
contributed as part of the OpenBEATs effort.
