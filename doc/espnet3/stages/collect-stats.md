---
title: ESPnet3 Collect Stats Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-14
---

# ESPnet3 Collect Stats Stage

`collect_stats` computes shape files and feature statistics used by later
training steps.
This serves two broad purposes:

1. **Shape information for batching**: Precomputing feature lengths lets the iterator adjust batches based on sequence size, which is one of the main ways ESPnet avoids out-of-memory errors. For more information on batching, see [Dataloader](../core/components/dataloader.html).
2. **Statistics for normalization**: Certain forms of normalization, such as global mean and variance normalization, need dataset-level statistics to be computed. Computing these once here allows them to be reused later.

Note that `collect_stats` only processes the dataset's `train` and `valid` splits; `test` is ignored.

## 1. Run

```bash
python run.py --stages collect_stats --training_config conf/training.yaml
```

## 2. Outputs

The collected information is saved to the following files:

```text
${stats_dir}/
├── train/
│   ├── feats_shape       # features shapes for batching
│   ├── feats_stats.npz   # features statistics for normalization
│   └── stats_keys
└── valid/
    ├── feats_shape
    ├── feats_stats.npz
    └── stats_keys
```

## 3. Model Requirements

The model must possess a `collect_feats()` method.
An implementation of this method exists by default for all models built on ESPnet tasks (e.g. ASR, TTS).

Custom models should provide a compatible implementation of the method with the following interface:

```collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]```

For features of variable length, the return dictionary should contain a matching `*_lengths` tensor containing the lengths for each sample in the feature tensor.

Example:
```python
class MyCustomModel
  def collect_feats(
      self,
      speech: torch.Tensor,
      speech_lengths: torch.Tensor,
      **kwargs,
  ) -> Dict[str, torch.Tensor]:
      feats, feats_lengths = self._extract_feats(speech, speech_lengths)
      return {"feats": feats, "feats_lengths": feats_lengths}
```

## 4. Configuration

The `collect_stats` stage is configured in the same `training.yaml` used for training.

At minimum, the `stats_dir` key must be set to the directionary where the output files will be dumped.
Components that require shape or stats files (e.g. `model`, `dataloader`) should point to the corresponding files in `${stats_dir}/train/` or `${stats_dir}/valid/` (Note that these paths are read-only; results are always written to `stats_dir`).

For more information, see [Training Configuration](../config/train_config.html).

::: warning ASR: `model.normalize` does not survive into a `train` stage run in the same process
`collect_stats()` removes `model.normalize` / `model.normalize_conf` from `training_config.model`
**in place** before building the trainer
([`espnet3/systems/base/training.py`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/training.py)),
and `training_config` is the same config object the `train` stage reuses afterwards. If `collect_stats`
and `train` run in the same `run.py` invocation (e.g. the default `--stages all`), the model is built
with the task's default normalizer instead of your configured `global_mvn`/`stats_file`, with no
warning. If your recipe sets `model.normalize`, run `collect_stats` and `train` as two separate
`run.py` invocations, and check the saved `${exp_dir}/config.yaml` after training to confirm
`normalize`/`normalize_conf` were actually applied.

`TTSSystem` does not have this problem: its `collect_stats` override
([`espnet3/systems/tts/system.py`](https://github.com/espnet/espnet/blob/master/espnet3/systems/tts/system.py))
builds the trainer without popping `normalize`/`normalize_conf`, because TTS's `normalize_choices`
defaults to `global_mvn` — popping the key would silently restore that default rather than disabling
normalization, and would also crash on the very first run (the stats file the default normalizer
expects does not exist yet).
:::

Example:

```yaml
stats_dir: ${exp_dir}/stats

# For shape-based batching
dataloader:
  train:
    iter_factory:
      batches:
        shape_files:
          - ${stats_dir}/train/feats_shape

# For normalization
model:
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
```

### Advanced: GPU-based stats collection

If `parallel` is configured in `training.yaml`, `collect_stats` can reuse ESPnet3's
parallel execution helpers for heavier feature extraction workloads.

Example:

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 30:00
    job_extra_directives:
      - "--gres=gpu:1"
```

## Related pages

**Stage API:** [`collect_stats`](../../guide/espnet3/components/collect_stats.html),
[`CollectStatsInferenceProvider`](../../guide/espnet3/components/CollectStatsInferenceProvider.html),
and [`DataLoaderBuilder`](../../guide/espnet3/components/DataLoaderBuilder.html).
Run this after [dataset preparation](./create-dataset.html) and before
[training](./train.html) when normalization or shape-based batching is enabled.

- [Training config](../config/train_config.html)

<DocCards :cols="3">
  <DocCard
    title="Train Config"
    desc="All available setting for training.yaml."
    icon="tabler:file-code"
    href="../config/train_config.html"
  />
</DocCards>
