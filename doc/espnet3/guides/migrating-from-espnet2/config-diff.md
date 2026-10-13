# ESPnet2 to ESPnet3 configuration differences

ESPnet3 separates the model, dataset, dataloader, inference, scoring, and
publication settings that ESPnet2 recipes often combine in YAML and shell
arguments. Model architecture settings can usually be reused, but batching
and execution settings need an explicit migration.

## Configuration mapping

| Setting | ESPnet2 | ESPnet3 |
| --- | --- | --- |
| Training config | `conf/train_asr_*.yaml`, selected by `--asr_config` in ASR recipes | `conf/training.yaml` or a variant, selected by `--training_config` |
| Inference config | `conf/decode_asr.yaml`, usually beam-search settings | `conf/inference.yaml`: model, dataset, provider/runner, outputs, and parallel execution |
| Scoring | Shell scoring stages | `conf/metrics.yaml`, selected by `--metrics_config` |
| Publication | Shell packing/upload options | `conf/publication.yaml`, selected by `--publication_config` |
| Demo | No direct config equivalent | `conf/demo.yaml`, selected by `--demo_config` |
| Batch sampler | `batch_type`, `batch_size`, `batch_bins` | `dataloader.<train or valid>.iter_factory.batches.type`, `batch_size`, `batch_bins` |
| Batch distribution | Sequence batches split across all ranks | `dataloader.<train or valid>.batch_distribution: per_gpu` by default; opt in to `global` for element-wise splitting |
| `batch_bins` semantics | Global budget for sequence batching | Per-GPU budget by default; global budget with `batch_distribution: global` |
| Gradient accumulation | `accum_grad` | `trainer.accumulate_grad_batches` |
| Epoch limit | `max_epoch` | `trainer.max_epochs` |
| Training parallelism | Distributed launch/shell options | Lightning settings under `trainer`; `num_device` selects devices per node |
| Helper/inference parallelism | `nj`, `run.pl`, `queue.pl` | Dask configuration under `parallel` |

## Preserve the intended batch size

In ESPnet2 sequence batching, a sampler produces a batch and each rank gets
`batch[rank::world_size]`. In ESPnet3's default `per_gpu` mode, each rank
gets complete batches using `batches[rank::world_size]`.

Consequently, copying an ESPnet2 `batch_bins` value into an ESPnet3 config
does not preserve the job-wide budget on multiple GPUs. With `per_gpu`,
the approximate budget per optimizer update scales with `world_size` and
gradient accumulation. Dynamic batch lengths and padding make this a budget,
not an exact sample count.

To retain ESPnet2 sequence-batching semantics, set the mode explicitly for
both loaders in `training.yaml`:

```yaml
dataloader:
  train:
    batch_distribution: global
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 4000000
        batch_size: 1
        min_batch_size: 2
  valid:
    batch_distribution: global
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/valid/feats_shape
        batch_bins: 4000000
        batch_size: 1
        min_batch_size: 2
```

This example requires every batch to contain at least `world_size` samples;
`min_batch_size: 2` is a starting point for two ranks, not a guarantee for
every sampler. Increase the minimum or revise the data/batch plan if necessary.
Uneven batches are split without padding or duplication. A batch smaller than
`world_size` raises an error; it is not silently dropped or duplicated.

Do not switch an existing ESPnet3 experiment to `global` merely to use a
new option: that changes its effective batch size. Omitting the option keeps
`per_gpu`, including existing whole-batch tail alignment. Switching modes
does not automatically rescale `batch_bins`, learning rate, or accumulation.
Record GPU count, mode, budget, and accumulation with the experiment.

Global mode requires a shared, unsharded dataset and the standard
SequenceIterFactory. Chunk/custom factories, rank-specific dataset sharding,
and `shuffle_within_batch: true` are rejected. See
[dynamic batching](../scaling/data-pipeline.md) for supported configurations.

## Model, optimizer, and trainer migration

Keep ESPnet2 architecture keys such as `encoder`, `encoder_conf`,
`decoder`, and `model_conf` under `model`, with a task bridge:

```yaml
task: espnet3.systems.esp2_asr.task.ASRTask
model:
  encoder: transformer
  encoder_conf:
    output_size: 256
  decoder: transformer
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 2500
trainer:
  accumulate_grad_batches: 1
  max_epochs: 200
```

An ESPnet3 model can instead be instantiated directly with `model._target_`.
Use the shipped recipe configs as the full reference; these examples show
only the sections being migrated.

## Dataset and path migration

Move source preparation into recipe `dataset/builder.py` and expose
`Dataset` and `DatasetBuilder` through `dataset/__init__.py`.
Configure DataOrganizer with `data_src` / `data_src_args` and named test
splits. Paths such as `recipe_dir`, `exp_dir`, `stats_dir`, and
`inference_dir` belong in the corresponding YAML, not host-specific shell
constants. See the [LibriSpeech reference recipe](../../../../egs3/librispeech_100/esp2_asr).

## Inference and metrics migration

Move decoding constructor settings such as `beam_size` and `ctc_weight`
under the inference config's `model`. The config also specifies datasets,
output directories, and execution settings. Training `batch_distribution`
does not change inference runner batching or `parallel.n_workers`.

For a declared InferenceAPI model, inputs and outputs come from its fields;
do not set `input_key`. An `output_fn` is normally unnecessary. For a
backend without that declaration, follow the runner's input/output contract.

Use the current ASR system path when configuring WER:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
```

Inference writes hypotheses; the measure stage can read reference text from
the dataset. Inspect the recipe's `metrics.yaml` for dataset/path settings.

The guides are linked by repository-relative paths until their official
documentation pages are published. No personal draft-site URL is required.
