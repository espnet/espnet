---
title: ESPnet3 Metrics
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Metrics

This page describes how to implement custom metrics for the current ESPnet3
measurement flow.

The base class is:

- `espnet3.components.metrics.base_metric.BaseMetric`

The stage entrypoint is:

- `espnet3.systems.base.metric.measure`

::: note
If you want the full file flow from `metrics.yaml` to `metrics.json`, use the
interactive explorer below.
This page focuses on the metric component contract itself.
:::

<MetricsExplorer />

## Metric contract

A metric subclasses `BaseMetric` and declares `inputs`/`outputs` as tuples
of `espnet3.contract.Field`: `inputs` names what it reads — the metrics
config's `inputs:` binds each declared name to a source — and `outputs`
names the result keys, each of kind `"number"`. There is no undeclared
fallback: a metric that declares neither raises `TypeError` as soon as it
is instantiated.

```python
from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric


class MyMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("score", "number"),)
```

A metric whose inputs/outputs never change (most of them — `WER`, `CER`,
`TER`) declares them as class attributes, as above. One whose contract
depends on its own configuration (a VERSA-style metric, say) sets
`self.inputs`/`self.outputs` in `__init__` before calling
`super().__init__()` instead.

`BaseMetric` itself implements the actual entry point `measure()` calls —
`__call__(data, test_name, output_dir)` — so a subclass does not override
`__call__`. It resets state, streams the declared inputs present in `data`
through `update()` in `chunk_size`-row batches, calls `compute()` once at
the end, and — only when `output_dir` is given — calls `write_artifacts()`:

```python
metric(data, test_name, output_dir)
# -> reset() -> update(**chunk) per chunk -> compute() -> write_artifacts()
```

`data` maps each declared input name to an SCP path (e.g. `ref.scp`,
`hyp.scp`); the metrics config's `inputs:` binds each name to its source
(see "Config aliases" below). `output_dir` is passed through as
`metrics_config.inference_dir` — in a typical Hydra/OmegaConf config this is
a plain `str`, not a `Path` instance; `write_artifacts` itself already
receives a proper `Path` (`output_dir / test_name`, already created).

**`metrics_config.inference_dir` must be set for `measure()` to run at all.**
It is populated when you pass `--inference_config` alongside
`--metrics_config` (the shared run copies `inference_config.inference_dir`
into `metrics_config`); passing only `--training_config` +
`--metrics_config` does not populate it, so set `inference_dir` explicitly in
`metrics.yaml` if you run `measure` standalone.

## BaseMetric

Implement `update`/`compute`, not `__call__`:

```python
from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric


class MyMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("score", "number"),)

    def update(self, *, ref, hyp):
        ...  # add one batch (equal-length sequences) to the state

    def compute(self):
        ...  # return {"score": ...} from the state alone
        return {"score": 0.0}
```

Return values must be JSON-serializable because `measure()` stores them in
`<inference_dir>/metrics.json`.

For a metric that accumulates state across `update()` calls (most of them),
register it with `self.add_state(name, default, reduce)` (`reduce="sum"`
for a running total, `"cat"` to concatenate lists) instead of a plain
instance attribute; `reset()`, `state()`, `merge_states()`, and
`load_state()` then work automatically, which matters for a metric also
used as a training callback across distributed ranks.

## iter_inputs() helper

`BaseMetric.__call__` already uses `iter_inputs()` internally to stream the
declared inputs into `update()` in `chunk_size`-row batches — a subclass
normally only needs `update`/`compute` and never calls it directly. Reach
for `iter_inputs()` yourself only when overriding `write_artifacts()` and
you need the raw per-utterance rows again (aligned SCP text inputs, read in
lockstep with utterance-ID checking):

```python
for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
    ref = row["ref"]
    hyp = row["hyp"]
```

## Config aliases

`inputs:` binds each name the metric *declares* (its `Field` names) to a
source: either a plain artifact stem, or `dataset:<column>` to read a
dataset column instead of an inference artifact.

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.CustomMetric
    inputs:
      ref: dataset:text
      hyp: hyp
      prompt: prompt
```

This means `measure()` passes:

- `data["ref"]` -> the test set's `text` dataset column
- `data["hyp"]` -> `hyp.scp`
- `data["prompt"]` -> `prompt.scp`

`inputs` can also be a plain list of names when each one is its own source
(e.g. `inputs: [ref, hyp]`). Every non-optional declared `Field` must be
bound; an unbound optional one is simply absent from `data`.

## Multiple entries of the same metric class

`measure()` keys its results dict by the metric's fully-qualified class path
(e.g. `espnet3.systems.esp2_asr.metrics.wer.WER`). If you configure the same class
twice in `metrics.yaml` (for example, `WER` once against `hyp.scp` and again
against `hyp_nbest0.scp` with different `inputs`), the second entry overwrites
the first under that same key in `metrics.json`, and both write to the same
alignment filename (e.g. `wer_alignment`). Give each entry a distinguishable
metric class (a small subclass is enough) if you need more than one scored
entry of the same metric type to survive.

## TER and `bpemodel`

`TER` needs a trained SentencePiece model to tokenize text before scoring, so
point `bpemodel` at the path your recipe's tokenizer stage actually wrote to —
typically `${tokenizer.save_path}/bpe.model`, which (per the TEMPLATE
tokenizer config) resolves to something like `${data_dir}/bpe_5000/bpe.model`,
not an `exp/` path.

## Example: text metric

```python
from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric
import jiwer


class SimpleWER(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("WER", "number"),)

    def __init__(self):
        self.add_state("refs", [], "cat")
        self.add_state("hyps", [], "cat")
        super().__init__()

    def update(self, *, ref, hyp):
        self.refs.extend(ref)
        self.hyps.extend(hyp)

    def compute(self):
        return {"WER": jiwer.wer(self.refs, self.hyps) * 100}
```

## Example: counting metric

A metric that only needs a running total uses a `"sum"` state instead of
accumulating rows:

```python
from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric


class HypothesisCount(BaseMetric):
    inputs = (Field("hyp", "text"),)
    outputs = (Field("num_hypotheses", "number"),)

    def __init__(self):
        self.add_state("count", 0, "sum")
        super().__init__()

    def update(self, *, hyp):
        self.count += sum(1 for line in hyp if line.strip())

    def compute(self):
        return {"num_hypotheses": float(self.count)}
```

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Metrics configuration"
    desc="See how metric classes and inputs are selected from YAML."
    icon="tabler:settings-2"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Metrics stage"
    desc="Return to the stage-level metrics flow."
    icon="tabler:route"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Components overview"
    desc="Return to the full component map."
    icon="tabler:puzzle"
    href="./index.html"
  />
</DocCards>
