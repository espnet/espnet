---
title: ESPnet3 Measure Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-22
---

# ESPnet3 Measure Stage

This page describes the model evaluation flow in ESPnet3.

## 1. Run

`measure` scores files written by the `infer` stage, so it needs to know
where those files are. Pass `--training_config` and `--inference_config`
alongside `--metrics_config` so `metrics_config.inference_dir` is inherited
from `inference.yaml`'s own `inference_dir` automatically:

```bash
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

::: important
Passing only `--training_config` and `--metrics_config` (without
`--inference_config`) does **not** set `metrics_config.inference_dir` and
raises `omegaconf.errors.ConfigAttributeError: Missing key inference_dir`.
Stage configs inherit top-level keys from every earlier-declared stage's own
config (`inherit`/`build_stage_configs` in
[`espnet3/utils/stage_configs.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/stage_configs.py)),
not just the directly preceding one -- `measure`'s config inherits
`inference_dir` from `infer`'s own config (`inference.yaml` declares
`inference_dir: ${exp_dir}/${self_name:}` as one of its own top-level keys),
and `infer`'s own config only exists when `--inference_config` is given.
`training.yaml` has no `inference_dir` key at all, so passing only
`--training_config` never supplies it. Always pass `--inference_config` when
running `--stages measure`.
:::

For a standalone `measure` run with no training/inference config, set
`exp_tag` (or a concrete `exp_dir`) and `inference_dir` directly in
`metrics.yaml`.

## 2. Outputs

The summary file format is:

```text
<inference_dir>/
├── metrics.json
└── test-clean/
    ├── ref.scp
    └── hyp.scp
```

`metrics.json` is keyed by metric class path, then by test set name.
`measure()` resolves test sets in this order:

1. If `metrics_config.dataset.test` exists, use each item's `name`.
2. Otherwise, scan `metrics_config.inference_dir` for subdirectories.

## 3. Configuration

The `measure` stage is configured using `metrics.yaml`
This config file defines the metrics that are used to evaluate the model.

Each entry in `metrics.yaml` is handled like this:

1. instantiate `metrics_config.metrics[*].metric`
2. resolve input SCP paths for one `test_name`
3. call the metric class

Example config for WER:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER # import path for the function
    inputs:
      ref: ref
      hyp: hyp
```

`WER` declares `inputs = (Field("ref", "text"), Field("hyp", "text"))` (see
[Metric contract](../core/components/metrics.html#metric-contract)); the
`inputs:` mapping above binds each declared name to the SCP file stem
written during the `infer` stage (here, each name to a same-named stem).

`measure()` will instantiate the class provided in `_target_` and pass the arguments as follows:

```python
{
    "ref": Path("exp/.../inference/<test_name>/ref.scp"),
    "hyp": Path("exp/.../inference/<test_name>/hyp.scp"),
}
```
where `<test_name>` is the name of the test partition (e.g., `test-clean`).

::: important
`measure()` does not preload SCP contents into lists. It resolves file paths and passes them directly to each metric.
:::

### Inputs and SCP filenames

`inputs:` in config binds each name a metric *declares* (its `Field` names)
to a source: either a plain SCP stem, or `dataset:<column>` to read a dataset
column instead:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.MyMetric
    inputs:
      ref: ref
      hyp: hyp
      prompt: prompt
```

then ESPnet3 resolves:

- `data["ref"] -> <test_name>/ref.scp`
- `data["hyp"] -> <test_name>/hyp.scp`
- `data["prompt"] -> <test_name>/prompt.scp`

A metric that declares `inputs` (as `WER`/`CER` do) requires a matching
`inputs:` binding in config -- omitting it raises `MetricContractError("<class> declares inputs; bind them in the metrics config")`
([`espnet3/contract/metrics.py`](https://github.com/espnet/espnet/blob/master/espnet3/contract/metrics.py)'s
`check_metric_inputs`). Only a metric that declares no inputs at all can omit
`inputs:` in config.


### Sample config

```yaml
recipe_dir: .
# exp_tag / exp_dir / inference_dir are normally propagated automatically
# from --training_config / --inference_config (see Section 1). Set them
# here only for a standalone `measure` run.
exp_tag:
exp_dir: ${recipe_dir}/exp/${exp_tag}

metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
      clean_types:
    inputs:
      ref: ref
      hyp: hyp

  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.cer.CER
      clean_types:
    inputs:
      ref: ref
      hyp: hyp
```


## 4. Custom Metrics

To implement a custom metric, create a class that inherits from
`espnet3.components.metrics.base_metric.BaseMetric`, declare `inputs`/
`outputs` as tuples of `espnet3.contract.Field`, and implement `update`/
`compute`:

```python
from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric


class MyMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("WER", "number"),)

    def update(self, *, ref, hyp):
        ...  # add one batch (equal-length sequences) to the state

    def compute(self):
        ...  # return {"WER": ...} from the state alone
```

`BaseMetric` itself implements `__call__(data, test_name, output_dir)` — the
entry point `measure()` actually calls — so a subclass does not override
`__call__`. It resolves `data` (a mapping from each declared input name to
its SCP path) into rows via `iter_inputs()`, streams them into `update()` in
batches, calls `compute()` once at the end, and writes any declared
artifacts. A metric that accumulates values across `update()` calls should
register that state with `self.add_state(name, default, reduce)`
(`reduce="sum"` or `"cat"`) so `reset()`/`state()`/`merge_states()`/
`load_state()` work for it automatically — needed when the same metric is
also used as a training callback across distributed ranks. See the
[Metrics component guide](../core/components/metrics.html) for the full
contract and worked examples.



## Related pages

**Stage API:** [`measure`](../../guide/espnet3/systems/measure.html),
[`BaseMetric`](../../guide/espnet3/components/BaseMetric.html),
[`WER`](../../guide/espnet3/systems/WER.html), and
[`CER`](../../guide/espnet3/systems/CER.html). This stage consumes files from
[inference](./inference.html) and can precede [publication](./publish.html).

<DocCards :cols="3">
  <DocCard
    title="Custom metrics"
    desc="Learn how to implement custom evaluation metrics."
    icon="tabler:file-code"
    href="../core/components/metrics.html"
  />
</DocCards>
