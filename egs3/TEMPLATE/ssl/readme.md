# ESPnet3 BEATs self-supervised pre-training recipe

Shared defaults for BEATs iterative pre-training
([arXiv:2212.09058](https://arxiv.org/abs/2212.09058)) with
`espnet3.systems.ssl.system.BeatsSystem`. Real recipes (for example
`egs3/audioset/ssl`) import `run.py` from here and keep only overrides in their
own `conf/`.

## Pipeline

One `run.py` invocation trains one BEATs iteration, selected by
`iteration` in the training config.

| Stage | Config | Iteration 0 | Iteration N > 0 |
|---|---|---|---|
| `create_dataset` | `--training_config` | Build the recipe manifests | (already built) |
| `train_tokenizer` | `--train_tokenizer_config` | No-op | Distill a VQ tokenizer from `teacher_ckpt_path`; export `beats_tokenizer_iter<N>.pt` |
| `infer` | `--inference_config` | Random-projection tokenization | Tokenize with `beats_tokenizer_iter<N>.pt` |
| `collect_stats` | `--training_config` | Write `feats_shape` | (reuse, stats are iteration-invariant) |
| `train` | `--training_config` | Train the encoder; export `beats_encoder_iter<N>.pt` | Same |

Outputs of iteration `N` with `ssl_tag: base`:

```
exp/stats_base/{train,valid}/feats_shape
exp/beats_iter<N>_base/targets/{train,valid}/{target.scp,target_shape}
exp/beats_iter<N>_base/beats_encoder_iter<N>.pt
exp/beats_tokenizer_iter<N>_base/beats_tokenizer_iter<N>.pt   # N > 0
```

## Quick start

```bash
# Iteration 0
python run.py --stages create_dataset infer collect_stats train \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# Iteration 1 (`training_iter1.yaml` sets `iteration: 1` and
# `teacher_ckpt_path` to the iteration-0 encoder)
python run.py --stages train_tokenizer infer train \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml
```

Multi-GPU training runs every stage of the invocation in each rank, so run
`train_tokenizer` and `train` in their own `run.py` invocations when
`num_device > 1`.

## Notes for recipe authors

- `fbank_mean`, `fbank_std`, `waveform_input`, `iteration`, and
  `teacher_ckpt_path` are set once in the training config; `run.py` copies them
  into the tokenizer training and inference configs.
- `dataset.test[*].name` in `inference.yaml` must match the target directories
  read by the training config (`${target_dir}/<name>/target.scp`), and each
  entry must select the same items in the same order as the corresponding
  training entry, because targets are keyed by dataset index.
- `batch_bins` is per device. ESPnet2 splits a batch across GPUs, so an egs2
  `batch_bins` on N GPUs corresponds to `batch_bins / N` here.
