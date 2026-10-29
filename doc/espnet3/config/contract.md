---
title: ESPnet3 Contracts
---

# ESPnet3 Contracts

Contracts describe fields and stage interfaces. They do not replace dataset
construction, `input_key`, backend implementation, or metric source bindings.
The relevant namespace is **`espnet3.contract`** (singular).

| Boundary | Declaration | Checked behavior |
| --- | --- | --- |
| Recipe dataset | `Dataset.fields` | sample keys and kinds before preprocessing |
| Inference | `contract.inputs` / `contract.outputs` in inference YAML | shared runner input/output validation and normalization |
| Metric | metric class/instance `inputs` / `outputs` | input batches and numeric result keys |
| System stages | class `stages` tuple of `StageSpec` | stage methods, order, roles and log directories |

## `Field` and built-in kinds

A Python declaration uses `Field(name, kind, label="", optional=False,
channels=None, options=...)`. The first positional argument is the field's
**name**, and the second is its registered **kind**, not a shape or Python
type. Prefer keyword arguments for the remaining settings:

```python
from espnet3.contract import Field


speech = Field("speech", "audio", options={"sample_rate": 16000})
text = Field("text", "text", label="Transcription")
```

The baseline registers these four kinds:

| Kind | Checked representation |
| --- | --- |
| `audio` | `Audio(array, rate)`, float32 mono or channels-first waveform |
| `text` | Python `str`, including the empty string |
| `number` | real scalar, including NumPy real scalars; excludes booleans |
| `segments` | list of mappings containing `text`, `start`, and `end` |

`segments` currently checks required keys, not a complete timestamp schema.
There is no built-in vector/image/reference-audio kind in this implementation;
register a suitable custom kind when needed. Having a contract kind does not
by itself install a demo UI asset or offline artifact serializer.

## Inference YAML

Put input/output declarations inside the top-level `contract:` block of
**`conf/inference.yaml`**:

```yaml
input_key:
  - speech
  - prompt

contract:
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      channels: 1
      normalize:
        - first_channel
    prompt:
      kind: text
      optional: true
  outputs:
    text:
      kind: text
      label: Transcription
      normalize:
        - strip
        - to_lower
```

`inputs` and `outputs` must be nonempty mappings. Common descriptor keys are
`kind`, `label`, `optional`, and `channels`; other keys are validated as
kind-specific options. Unknown top-level contract keys, unknown kinds, and
unsupported kind options raise errors while loading. Outputs cannot be
optional. `input_key` must contain exactly the input names, with no duplicates.
An optional input must still appear in `input_key`; it may be absent or `None`
in samples, but cannot have mixed presence in one batch.

The runner validates named output mappings **after** `format_output()`.
Every declared output is required; extra keys other than the identifier are
rejected. The runner adds `utt_id` (or the configured `idx_key`) from the
sample/index. The identifier is not a contract output field.

`infer`, `InferenceModel`, and the demo all call the configured runner's
`forward()`. The generic runner passes checked inputs through its
`prepare_model_inputs(inputs)` hook unchanged. `ASRInferenceRunner` adapts
`Audio` inputs and ESPnet2 n-best outputs. A compatible backend can use the
generic runner directly; subclass only to adapt behavior. Legacy `output_fn`
and a contract are mutually exclusive.

## Audio checking and normalization

Audio accepts an `Audio`, a path, a `(rate, waveform)` pair, a NumPy array, or
a CPU tensor. Arrays with no rate are assumed to use the field's declared
rate (or the model's rate where available). Pass a pair or `Audio` when the
actual source rate matters:

```python
import numpy as np

from espnet3.contract.kinds.audio import Audio


reference = Audio(np.zeros(8000, dtype=np.float32), rate=8000)
resampled = reference.resample(16000)
assert resampled.rate == 16000
```

Audio options:

| Option | Meaning |
| --- | --- |
| `sample_rate` | desired sample rate |
| `rate_policy` | `resample` (default) or `reject` on a known-rate mismatch |
| `channels` | exact required channel count; default `None` preserves channels |
| `normalize` | ordered normalizer names |

Samples must be real and finite; NaN/Inf are rejected. Integer PCM is scaled
to float32. The internal layout is `(T,)` for mono or `(channels, T)` for
multichannel input. The converter recognizes typical samples-first audio;
ambiguous very short arrays keep channels-first layout.

`channels: 1` is a constraint, not automatic downmixing. Use
`first_channel` or `mean_channels` explicitly when a mono backend should
accept stereo input. Normalization runs after initial checking, followed by
another check, so the final channel requirement applies after normalization.
Text normalizers are `strip`, `to_lower`, and `to_upper`.

```yaml
contract:
  inputs:
    speech:
      kind: audio
      sample_rate: 16000
      rate_policy: reject
      channels: 1
      normalize:
        - mean_channels
  outputs:
    text:
      kind: text
```

## Custom kinds and normalizers

Place registration code in a recipe module and import it through
`contract.imports`. Use absolute Python imports:

```python
# src/contracts.py
import numpy as np

from espnet3.contract import Kind, register_kind, register_normalizer
from espnet3.contract.kinds.audio import Audio


class SpectrogramKind(Kind):
    def check(self, value, field, model, *, output):
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise TypeError(f"{field.name} must be a 2-D NumPy array")
        return value


def halve_volume(audio):
    return Audio(audio.array * 0.5, audio.rate)


register_kind("spectrogram", SpectrogramKind())
register_normalizer("halve_volume", halve_volume)
```

```yaml
contract:
  imports:
    - src.contracts
  inputs:
    features:
      kind: spectrogram
  outputs:
    score:
      kind: number
```

Each normalizer receives the checked representation and returns a value of
the same kind. Global registration names must be unique unless `replace=True`
is explicitly passed. A custom kind may implement `to_model_input()` to adapt
its checked representation, `to_demo_output()` for display, and `join()` for
streamed pieces. These hooks do not make an arbitrary backend streaming.

Packed models retain the contract in their inference YAML. Loading bundled
registration modules requires `InferenceModel.from_packed(...,
trust_user_code=True)` or `from_pretrained(..., trust_user_code=True)`;
a demo forwards `model.trust_user_code`. A custom kind also needs a registered
UI asset when its kind name is used as a UI type. Only audio/text assets are
registered by default on this baseline.

## Dataset declaration belongs on the class

```python
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field


class Dataset(BaseDataset):
    fields = (
        Field("speech", "audio"),  # name, registered kind
        Field("text", "text"),
    )

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
```

Pass the raw sample mappings to this dataset constructor.
Declare the raw fields actually returned, not post-preprocessor tensors. The
base dataset checks the first item returned by `__getitem__`; extra keys
are permitted, missing required fields are not. This does not promise a
full-dataset scan or duplicate-identifier checking. Optional fields may be
omitted from an individual item's declaration check.

## Metric declaration and YAML bindings are separate

A metric declares its interface on the **metric class**, or on the instance
before `super().__init__()` for configuration-dependent fields:

```python
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class ExampleMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("score", "number"),)

    def update(self, ref, hyp):
        pass  # replace with accumulation of per-sample metric state

    def compute(self):
        return {"score": 0.0}  # replace with the computed numeric score
```

Every output kind must be `number`. Bind declared input names on the
**individual metric entry** in `conf/metrics.yaml`, alongside `metric:`:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
```

This reads the model's `text.scp` and the test dataset's `text` column.
`inputs` does not belong at the config root or inside `metric:` constructor
kwargs. Results are written to `${inference_dir}/metrics.json`.

## Stage contracts

Systems declare an ordered tuple of `StageSpec(name=..., config=...,
log_dir=...)`. `config` names one role or a tuple of roles; it is not a file
path. Add custom stages by constructing a full tuple with inherited stages
plus your additions, and define the corresponding method. No automatic
stage-name-only insertion API is provided. See the
[configuration/launch reference](./index.html) for roles and the reviewed
launcher's predecessor-entry limitations.
