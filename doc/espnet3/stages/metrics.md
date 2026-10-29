# Measure stage

`measure` binds each metric's declared inputs to inference SCPs or test-dataset
columns, scores the test sets, and writes `metrics.json`.

## Run through the CLI

```bash
python run.py --stages infer measure \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

The current launcher retains only requested stage entries, and `BaseSystem.measure`
reads `self.stage_configs["infer"]`. Include `infer` in the CLI stage list; passing
`--inference_config` with `--stages measure` alone does not retain that entry.
Inference's completed shards are reused by default when its shard plan is unchanged.
For scoring without invoking inference at all, construct a system directly, as in
the [publication example](./publish.html#pack-without-rerunning-training), select
`stages_to_run=("measure",)`, and call `system.measure()`.

## Metric input bindings

Use a recipe `conf/metrics.yaml` overlay such as:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.cer.CER
    inputs:
      ref: dataset:text
      hyp: text
```

`inputs` is a sibling of `metric`, not part of the metric constructor. WER and CER
have no `ref_key`/`hyp_key` constructor options. Their declared names are `ref`
and `hyp`; the bindings say where those values come from:

| Binding | Source |
| --- | --- |
| `hyp: text` | `<inference_dir>/<test_name>/text.scp` |
| `ref: dataset:text` | the test dataset's `text` column |

A metric declaring inputs requires explicit bindings. Missing required names,
unknown names, and empty source strings are rejected. The resolver passes
`dict[str, Path]` to the metric; `BaseMetric.__call__` loads aligned SCP values in
chunks and supplies named value lists to `update`. It does not pass raw file paths
to `update`.

A dataset-column source is materialized on first use under
`<inference_dir>/<test_name>/dataset/<column>.scp`. A scalar is stored inline;
non-scalar values use artifacts beside that SCP. A second measurement reuses this
file: remove it when the dataset column changes. Audio reference columns can use:

```yaml
dataset_artifacts:
  speech:
    type: wav
    sample_rate: 16000
```

`dataset.test[*].name`, when configured in metrics, explicitly selects test sets.
Otherwise the stage scans non-hidden subdirectories of `inference_dir`. Use an
explicit list if that directory also contains stale or unrelated subdirectories.

## Declare a custom metric

Implement `update` and `compute` on a `BaseMetric` subclass. `inputs`/`outputs`
belong inside the class, or on `self` before `super().__init__()` for a
configuration-dependent contract. Every metric output must have kind `number`.
Here `Field`'s first positional argument is the field name and the second is the
registered kind:

```python
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class ExactMatch(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("accuracy", "number"),)

    def __init__(self):
        super().__init__()
        self.add_state("correct", 0, "sum")
        self.add_state("total", 0, "sum")

    def update(self, *, ref, hyp):
        self.correct += sum(a == b for a, b in zip(ref, hyp))
        self.total += len(ref)

    def compute(self):
        return {"accuracy": self.correct / self.total if self.total else 0.0}
```

Store reusable implementation under `espnet3/systems/<name>/metrics/`, or
recipe-local code under `src/`, and point `metric._target_` to its absolute module
path. The base call resets state for each test set, validates aligned batches,
and writes optional metric artifacts. Override `write_artifacts(test_dir)` for
alignment or diagnostic files; ordinarily do not replace `__call__`.

## Results

`metrics.json` is keyed by metric class path, then test-set name, then declared
result key. ASR's outputs are uppercase `WER` and `CER` percentages; for example:

```json
{
  "espnet3.systems.esp2_asr.metrics.wer.WER": {
    "test": {
      "WER": 4.3
    }
  }
}
```

Only one instance per metric class path can be represented safely in the current
result mapping; later instances of the same class overwrite earlier results.
Metric inputs and numeric results are checked independently of the inference
model's contract.

## Related sources

- [Measurement and dataset-column resolution](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/metric.py)
- [BaseMetric](https://github.com/espnet/espnet/blob/master/espnet3/components/metrics/base_metric.py)
- [Metric contracts](https://github.com/espnet/espnet/blob/master/espnet3/contract/metrics.py)
- [Inference](./inference.html) and [publication](./publish.html)
