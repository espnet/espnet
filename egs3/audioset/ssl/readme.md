# BEATs pre-training on AudioSet-2M (ESPnet3)

ESPnet3 port of [`egs2/audioset/ssl1`](../../../egs2/audioset/ssl1): iterative
[BEATs](https://arxiv.org/abs/2212.09058) pre-training of a Transformer audio
encoder together with a VQ acoustic tokenizer, following the paper's Table 4
setup. Stages are implemented by `espnet3.systems.ssl.system.BeatsSystem`; the
shared defaults live in [`egs3/TEMPLATE/ssl`](../../TEMPLATE/ssl).

## Setup

Set `AUDIOSET` (or `create_dataset.source_dir` in `conf/training.yaml`) to the
AudioSet root, which must contain `{eval,balanced_train,unbalanced_train}_segments.csv`
and the downloaded clips in `eval_wav/`, `balance_wav/`, and `unbalanced_wav/`
(16 kHz mono wav). We recommend installing
[Flash Attention](https://github.com/Dao-AILab/flash-attention) and training on
Ampere-or-newer GPUs (`bf16-mixed`).

## Run

Each `run.py` invocation trains one BEATs iteration. With `num_device > 1`,
run the GPU training stages (`train_tokenizer`, `train`) in their own
invocations, because Lightning re-runs the invoked stages in every rank.

```bash
. ./path.sh
export AUDIOSET=/path/to/audioset

# Iteration 0: manifests, random-projection targets, shape stats
python run.py --stages create_dataset infer collect_stats \
    --training_config conf/training.yaml --inference_config conf/inference.yaml
# Iteration 0: encoder (writes exp/beats_iter0_base_as2m/beats_encoder_iter0.pt)
python run.py --stages train --training_config conf/training.yaml

# Iteration 1: tokenizer distilled from the iteration-0 encoder
python run.py --stages train_tokenizer \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml
# Iteration 1: tokenize with it, then re-train the encoder
python run.py --stages infer \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml
python run.py --stages train --training_config conf/training_iter1.yaml
```

The exported `beats_encoder_iter<N>.pt` loads directly into the downstream
classification recipes (`egs2/esc50/asr1`, `egs2/as20k/cls1`) via
`beats_ckpt_path`. `BeatsEncoder` does not read fbank statistics from the
checkpoint, so also set the pre-training statistics in the downstream
`encoder_conf` (not inside `beats_config`):

```yaml
encoder_conf:
    beats_ckpt_path: /path/to/beats_encoder_iter1.pt
    fbank_mean: 15.66439
    fbank_std: 6.38312
```

## Mapping from egs2/audioset/ssl1

| egs2 | ESPnet3 |
|---|---|
| stages 1-3: `local/data.sh`, format, `--max_wav_duration 11` | `create_dataset` (`dataset/builder.py`) |
| stage 4: fbank dump + corpus stats | fbank computed on the fly from waveforms (`waveform_input: true`); `fbank_mean`/`fbank_std` in `conf/training.yaml` |
| stage 5: `audio_tokenization.sh` (`beats_random` / `beats`) | `infer` (`BeatsTokenizationModel`) |
| stage 6: collect stats | `collect_stats` |
| stage 7: `train_tokenizer`, `tokenizer_inference`, `train_encoder`, `generate_checkpoint` | `train_tokenizer`, `infer`, `train` (checkpoint export included) |
| `conf/beats_base.yaml` + `conf/ds_beats.json` | `conf/training.yaml` (+ TEMPLATE defaults) |
| `conf/tok_beats_base.yaml` + `conf/ds_beats_tok.json` | `conf/training_tokenizer.yaml` (+ TEMPLATE defaults) |
| `conf/as2m_inf.yaml` | `conf/inference.yaml` (+ TEMPLATE defaults) |

Differences from the egs2 recipe:

- `batch_bins` is per device in ESPnet3 (ESPnet2 splits each batch across GPUs),
  so egs2's `batch_bins: 420000` on 2 GPUs is `210000` per device here.
- DeepSpeed (ZeRO stage 0, bf16) is replaced by Lightning DDP with
  `bf16-mixed`; `WarmupDecayLR` by the equivalent `PiecewiseLinearWarmupLR`.
- `fbank_mean`/`fbank_std` are set once in `conf/training.yaml` and passed as
  `encoder_conf` arguments to the encoder, the tokenizer, the tokenizer's
  teacher, and tokenization alike. In egs2, stage 4 writes the corpus
  statistics into the configs correctly, but in the encoder and tokenizer
  configs they sit inside `beats_config`/`tokenizer_config`. `BeatsEncoder`
  only takes them as constructor arguments, so those values are stored but
  never used: encoder and tokenizer training (and checkpoints loaded as a
  teacher) normalized with the library defaults (15.29130/5.90532). Only
  iteration-0 tokenization, whose `conf/as2m_inf.yaml` has the values at the
  top level, used the AudioSet statistics. The released egs2 checkpoints were
  therefore trained with the defaults, which is also what downstream recipes
  use when `fbank_mean`/`fbank_std` are not set.
- Existing egs2 fbank dumps can be reused with
  `data_src_args.feats_path: <egs2>/dump/fbank/<split>/feats.scp` and
  `waveform_input: false`.

## Pretrained models (egs2 recipe)

| Model | Link |
|---|---|
| BEATs encoder, iter 0 | [jaeyeonkim99/BEATs-base-AS2M-iter0](https://huggingface.co/jaeyeonkim99/BEATs-base-AS2M-iter0) |
| BEATs encoder, iter 1 | [jaeyeonkim99/BEATs-base-AS2M-iter1](https://huggingface.co/jaeyeonkim99/BEATs-base-AS2M-iter1) |
| BEATs tokenizer, iter 1 | [jaeyeonkim99/BEATs-tokenizer-AS2M-iter1](https://huggingface.co/jaeyeonkim99/BEATs-tokenizer-AS2M-iter1) |

# RESULTS

Full AS-2M pre-training with this ESPnet3 recipe is in progress. Downstream
results of the egs2 recipe (AS-20K test mAP / ESC-50 5-fold accuracy) are
29.72 / 93.35 for iteration 0 and 31.26 / 93.35 for iteration 1.

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
