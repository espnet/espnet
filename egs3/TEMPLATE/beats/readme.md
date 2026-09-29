# ESPnet3 BEATs pre-training template

Shared defaults for BEATs iterative pre-training
([arXiv:2212.09058](https://arxiv.org/abs/2212.09058)) with
`espnet3.systems.beats.system.BeatsSystem`. Real recipes (for example
`egs3/audioset/beats`) import `run.py` from here and keep only overrides in their
own `conf/`.

## Pipeline

One `run.py` invocation trains one BEATs iteration, selected by
`iteration` in the training config. `pretrain` runs the first five stages
below in order on a single device; the defaults (`--stages all`) are
`pretrain measure pack_model upload_model`.

| Stage | Config | Iteration 0 | Iteration N > 0 |
|---|---|---|---|
| `create_dataset` | `--training_config` | Build the recipe manifests | (already built) |
| `train_tokenizer` | `--train_tokenizer_config` | No-op | Distill a VQ tokenizer from `teacher_ckpt_path`; export `beats_tokenizer_iter<N>.pt` |
| `infer` | `--inference_config` | Random-projection tokenization | Tokenize with `beats_tokenizer_iter<N>.pt` |
| `collect_stats` | `--training_config` | Write `feats_shape` | (reuse, stats are iteration-invariant) |
| `train` | `--training_config` | Train the encoder; export `beats_encoder_iter<N>.pt` | Same |
| `measure` | `--metrics_config` | Codebook usage of the targets (`CodebookUsage`) | Same |
| `pack_model` / `upload_model` | `--publication_config` | Bundle / upload the exported encoder | Same |

The `.pt` exports are written by the `BeatsCheckpointExport` callback
(`trainer.callbacks` in both training configs) when training ends, from the
top-K checkpoints of that run.

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
python run.py --stages pretrain measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# Iteration 1 (`training_iter1.yaml` sets `iteration: 1` and
# `teacher_ckpt_path` to the iteration-0 encoder)
python run.py --stages pretrain measure \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Multi-GPU training runs every stage of the invocation in each rank, so
`pretrain` refuses `num_device * num_nodes > 1`. Run `create_dataset`,
`train_tokenizer`, `infer`, `collect_stats`, and `train` as separate `run.py`
invocations instead (see `egs3/audioset/beats/readme.md`).

## Notes for recipe authors

- `fbank_mean`, `fbank_std`, `waveform_input`, `iteration`, and
  `teacher_ckpt_path` are set once in the training config; `run.py` copies them
  into the tokenizer training and inference configs, where they are `???`.
- The TEMPLATE values are the BEATs base model (paper Table 4) with a 400K-step
  budget (`trainer.max_steps`). Recipes that count in epochs of their corpus
  (e.g. AudioSet-2M: 56 encoder / 14 tokenizer epochs) set `max_epochs` and
  `max_steps: -1`.
- `dataset.test[*].name` in `inference.yaml` must match the target directories
  read by the training config (`${target_dir}/<name>/target.scp`), and each
  entry must select the same items in the same order as the corresponding
  training entry, because targets are keyed by dataset index.
- `batch_bins` is per device. ESPnet2 splits a batch across GPUs, so an egs2
  `batch_bins` on N GPUs corresponds to `batch_bins / N` here.
