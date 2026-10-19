---
title: ESPnet3 Inference Contract
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Inference Contract

The inference contract is a declarative description of a model's inputs and
outputs, including their types, constraints, and optional normalizations. It is
specified in the `contract` section of `inference.yaml` and is used by the
inference runner to validate inputs, transform data, and validate outputs.

## Overview

A contract consists of:

- `inputs`: a mapping from input field names to field descriptors
- `outputs`: a mapping from output field names to field descriptors
- `imports` (optional): a list of Python modules that register custom kinds
  and/or normalizers

Example:

```yaml
contract:
  imports:
    - src.contracts  # optional custom kinds/normalizers
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      rate_policy: resample
      channels: 1
      normalize: [mean_channels]
    prompt:
      kind: text
      label: Optional Prompt
      optional: true
  outputs:
    text:
      kind: text
      label: Transcription
      normalize: [strip, to_lower]
```

## Field Descriptor

Each field descriptor is a YAML mapping with the following keys:

| Key | Required | Description |
| --- | --- | --- |
| `kind` | yes | The data kind: `audio`, `text`, `number`, or `segments` |
| `label` | no | Human-readable label for UI and documentation |
| `optional` | no | If `true`, the field is optional (default `false`) |
| `channels` | no | For `audio`: exact channel count required by the model |
| `sample_rate` | no | For `audio`: target sample rate (Hz) |
| `rate_policy` | no | For `audio`: `resample` (default) or `reject` mismatched rates |
| `normalize` | no | List of normalizers to apply (see below) |

Unknown keys raise a `ValueError` during contract loading.

## Built-in Kinds

### audio

An audio signal with optional sample rate and channel information.

Supported options:

- `sample_rate` (int): Target sample rate in Hz. If the input has a known
  rate, it is resampled to this value when `rate_policy: resample` (default).
  Use `rate_policy: reject` to refuse mismatched rates.
- `rate_policy` (`resample` | `reject`): Policy for handling sample rate
  mismatches (default `resample`).
- `channels` (int): Exact channel count required. The model receives an `Audio`
  object with all channels preserved unless this is set, in which case the
  channel count is validated. For stereo input with a mono model, add
  `normalize: [mean_channels]` or `[first_channel]`.
- `normalize` (list): Audio normalizers to apply before the model sees the
  input. Built-in options: `mean_channels`, `first_channel`.

Example:

```yaml
speech:
  kind: audio
  label: Input Audio
  sample_rate: 16000
  rate_policy: resample
  channels: 1
  normalize: [mean_channels]
```

### text

A text string.

Supported options:

- `normalize` (list): Text normalizers to apply. Built-in options:
  `strip`, `to_lower`.

Example:

```yaml
text:
  kind: text
  label: Transcription
  normalize: [strip, to_lower]
```

### number

A numeric value (int or float). No kind-specific options.

Example:

```yaml
score:
  kind: number
  label: Confidence Score
```

### segments

A list of time-aligned segments with start/end times.

No kind-specific options.

Example:

```yaml
segments:
  kind: segments
  label: Aligned Segments
```

## Normalizers

Normalizers are functions that transform values of a specific kind. They are
applied in the order listed.

### Built-in Audio Normalizers

- `mean_channels`: Average all channels into one (mono)
- `first_channel`: Take only the first channel

### Built-in Text Normalizers

- `strip`: Remove leading/trailing whitespace
- `to_lower`: Convert to lowercase

### Custom Normalizers

Custom normalizers can be registered via `register_normalizer` from
`espnet3.contract` and loaded through the `imports` list. A normalizer
receives the Kind's checked value and returns the same Kind.

Example registration:

```python
from espnet3.contract import register_normalizer

@register_normalizer("audio", "custom_normalizer")
def custom_audio_normalizer(audio):
    # Transform audio here
    return audio
```

## Custom Kinds

Custom kinds can be registered via `register_kind` from `espnet3.contract` and
loaded through the `imports` list.

Example registration:

```python
from espnet3.contract import Field, register_kind

@register_kind("custom_kind")
class CustomKind(Field):
    # Custom kind implementation
    pass
```

## Imports

The `imports` list loads Python modules that register custom kinds and/or
normalizers. Modules are imported in order, and their registrations are
applied before the contract is used.

```yaml
contract:
  imports:
    - src.contracts.my_custom_kinds
    - src.contracts.audio_normalizers
```

**Security note:** Loading modules from a published bundle requires
`trust_user_code: true` in the model configuration, as the bundle root is
added to `sys.path`.

## Input/Output Validation

- `input_key` in `inference.yaml` must match the names declared in
  `contract.inputs` exactly; no duplicates are allowed.
- Optional inputs are declared with `optional: true` in the contract and must
  also be present in `input_key`. Within one batch, all samples must either
  all have or all lack an optional input, otherwise a `KeyError` is raised:
  "Optional inputs cannot be mixed within one batch".
- The model output must match the contract's output field names; the runner
  validates this and produces SCP files named after the output fields.

## Legacy Compatibility

- The `output_fn` key cannot be combined with a contract; setting both raises
  `ValueError("output_fn cannot be combined with an inference contract")`.
- Older bundles may still use `output_fn` for output formatting.
- The ESPnet2 ASR system moved n-best text formatting from `output_fn` into the
  runner, so the ASR template no longer sets `output_fn`.

## API Reference

The following functions are exported by `espnet3.contract` for registering
custom kinds and normalizers:

- `register_kind(kind_name: str)`: Decorator to register a custom kind
- `register_normalizer(kind: str, name: str)`: Decorator to register a normalizer
  for a specific kind
- `load_inference_contract(path: str)`: Load and validate a contract from a YAML
  file

See the [API reference](../api-reference.html) for full details.

## Full Example: Audio-to-Text ASR

```yaml
contract:
  imports:
    - src.contracts.asr_normalizers  # optional custom normalizers
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      rate_policy: resample
      channels: 1
      normalize: [mean_channels]
    prompt:
      kind: text
      label: Optional Prompt
      optional: true
  outputs:
    text:
      kind: text
      label: Transcription
      normalize: [strip, to_lower]
    confidence:
      kind: number
      label: Confidence Score
```

## Related Pages

- [Inference configuration](./infer_config.html)
- [Inference stage](../stages/inference.html)
- [Demo configuration](./demo_config.html)
- [Publication configuration](./publish_config.html)
