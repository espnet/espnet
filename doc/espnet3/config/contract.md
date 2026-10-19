---
title: ESPnet3 Inference Contract
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Inference Contract

The **inference contract** is a declarative description of a model's inputs and outputs, including their types, constraints, and optional normalizations. It is defined in the `contract:` section of `inference.yaml` and is the recommended way to describe model behavior for inference, publication, and demo stages.

## Overview

A contract consists of:

- `inputs`: A mapping from input field names to field descriptors
- `outputs`: A mapping from output field names to field descriptors
- `imports` (optional): List of recipe modules that register custom kinds and normalizers

Each field descriptor specifies:

- `kind`: The data type (`audio`, `text`, `number`, `segments`)
- `label`: Human-readable description
- `optional`: Whether the field is optional (default: `false`)
- Kind-specific options (e.g., `sample_rate`, `channels`, `normalize`)

Unknown keys in the contract raise a `ValueError` during loading.

## Basic Structure

```yaml
contract:
  imports:
    - src.contracts  # optional: load custom kinds/normalizers
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

## Field Descriptors

### Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `kind` | string | Data type: `audio`, `text`, `number`, or `segments` |
| `label` | string | Human-readable description for UI and documentation |
| `optional` | boolean | Whether the field is optional (default: `false`) |

### Kind-Specific Options

#### Audio Fields

| Option | Type | Description |
|--------|------|-------------|
| `sample_rate` | integer | Expected sample rate (Hz) |
| `rate_policy` | string | How to handle rate mismatches: `resample` (default) or `reject` |
| `channels` | integer | Exact channel count required; omit to pass through unchanged |
| `normalize` | list | List of normalizers to apply: `mean_channels`, `first_channel` |

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

#### Text Fields

| Option | Type | Description |
|--------|------|-------------|
| `normalize` | list | List of normalizers to apply: `strip`, `to_lower` |

Example:
```yaml
text:
  kind: text
  label: Transcription
  normalize: [strip, to_lower]
```

#### Number Fields

Number fields have no additional options beyond the common fields.

Example:
```yaml
confidence:
  kind: number
  label: Confidence Score
```

#### Segments Fields

Segments fields represent time-aligned segments (e.g., for speech recognition with timestamps).

Example:
```yaml
segments:
  kind: segments
  label: Aligned Segments
```

## Built-in Kinds

ESPnet3 provides four built-in kinds:

1. **audio**: An audio signal with optional sample rate and channel information
2. **text**: A text string
3. **number**: A numeric value (float or int)
4. **segments**: A list of segments with start/end times

## Normalizers

Normalizers are functions that transform values of a specific kind. They run before validation and model execution.

### Built-in Normalizers

#### Audio Normalizers

- `mean_channels`: Averages all channels to mono
- `first_channel`: Takes only the first channel

#### Text Normalizers

- `strip`: Removes leading/trailing whitespace
- `to_lower`: Converts text to lowercase

### Custom Normalizers

Custom normalizers can be registered using `register_normalizer` from `espnet3.contract`:

```python
from espnet3.contract import register_normalizer

@register_normalizer("audio", "custom_normalizer")
def custom_audio_normalizer(audio):
    # Transform audio here
    return transformed_audio
```

Custom normalizers are loaded through the `imports` list in the contract:

```yaml
contract:
  imports:
    - src.contracts  # Loads custom normalizers from this module
```

**Security Note**: Loading custom code from a published bundle requires `trust_user_code: true` in the model configuration.

## Input/Output Matching

### Input Key Matching

The `input_key` in `inference.yaml` must match the input field names declared in `contract.inputs` exactly. No duplicates are allowed.

```yaml
# inference.yaml
input_key: [speech, prompt]  # Must match contract.inputs keys

# contract in inference.yaml
contract:
  inputs:
    speech:
      kind: audio
    prompt:
      kind: text
      optional: true
```

### Optional Inputs

Optional inputs are declared with `optional: true` in the contract and must also be present in `input_key`. Within one batch, all samples must either all have or all lack an optional input, otherwise a `KeyError` is raised:

```
KeyError: Optional inputs cannot be mixed within one batch
```

### Output Validation

The model output must match the contract's output field names. The runner validates this and produces SCP files named after the output fields.

## Audio Behavior

When an audio input is provided:

1. If the input has a known sample rate, it is automatically resampled to the declared `sample_rate` (using `rate_policy: resample`)
2. The model receives an `Audio` object with all channels preserved
3. If `channels: N` is set, the exact channel count is validated; otherwise, multichannel input passes through unchanged
4. For stereo input with a mono model, add `normalize: [mean_channels]` or `[first_channel]` to the input descriptor

Example with stereo rejection:
```yaml
speech:
  kind: audio
  label: Input Audio
  sample_rate: 16000
  rate_policy: reject  # Reject mismatched rates
  channels: 1          # Require exactly 1 channel
```

Example with mono conversion:
```yaml
speech:
  kind: audio
  label: Input Audio
  sample_rate: 16000
  channels: 1
  normalize: [mean_channels]  # Convert stereo to mono
```

## Legacy and Exclusions

### `output_fn` Exclusion

The `output_fn` configuration cannot be combined with a contract. Setting both raises:

```
ValueError: output_fn cannot be combined with an inference contract
```

The ESPnet2 ASR system moved n-best text formatting from `output_fn` into the runner, so the ASR template no longer sets `output_fn`.

### Contract Travels with Packed Models

The contract is stored in the packed `inference.yaml` within the model bundle. When loaded via `InferenceModel.from_pretrained()` or `from_packed()`, the contract is automatically applied for validation and processing.

## Registration API

ESPnet3 provides registration functions for extending the contract system:

```python
from espnet3.contract import register_kind, register_normalizer

# Register a new kind
register_kind("my_kind", MyKindValidator)

# Register a normalizer for a kind
register_normalizer("my_kind", "my_normalizer", my_normalizer_function)
```

These functions are available from the `espnet3.contract` module.

## Examples

### Audio-to-Text ASR

```yaml
contract:
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      rate_policy: resample
      channels: 1
      normalize: [mean_channels]
  outputs:
    text:
      kind: text
      label: Transcription
      normalize: [strip, to_lower]
```

### Multi-Modal Model

```yaml
contract:
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      channels: 1
    text_prompt:
      kind: text
      label: Text Prompt
      optional: true
  outputs:
    text:
      kind: text
      label: Generated Text
    confidence:
      kind: number
      label: Confidence Score
```

### Custom Kind with Normalizer

```python
# In src/contracts.py
from espnet3.contract import register_kind, register_normalizer

class SpectrogramKind:
    def validate(self, value):
        # Validation logic
        pass

register_kind("spectrogram", SpectrogramKind)

@register_normalizer("spectrogram", "normalize_db")
def normalize_to_db(spec):
    # Convert spectrogram to dB scale
    return spec_to_db(spec)
```

```yaml
# In inference.yaml
contract:
  imports:
    - src.contracts
  inputs:
    spectrogram:
      kind: spectrogram
      label: Input Spectrogram
  outputs:
    logits:
      kind: number
      label: Output Logits
```

## Related Pages

- [Inference Configuration](./infer_config.html) — Overview of inference configuration including the contract section
- [Inference Stage](../stages/inference.html) — How the contract is used during inference
- [Demo Configuration](./demo_config.html) — How contracts generate UI widgets
- [Publication Configuration](./publish_config.html) — How contracts travel with packed models
