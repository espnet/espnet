---
title: ESPnet3 Inference Configuration
---

# ESPnet3 Inference Configuration

Inference YAML configures a callable backend, the data passed to it, and the
runner that formats and persists results. Run from a recipe directory, such
as `egs3/mini_an4/esp2_asr`:

```bash
python run.py --stages infer \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml
```

The training role supplies experiment identity and other earlier-stage keys.
The inference role supplies its own model/dataset configuration through
[stage inheritance](./index.html#loading-and-inheritance). It is not a shared
training config object, and explicit inference identity keys are not discarded.

## TEMPLATE defaults versus recipe overrides

The ASR TEMPLATE is `egs3/TEMPLATE/esp2_asr/conf/inference.yaml`.

| Key | ASR TEMPLATE value or behavior |
| --- | --- |
| `inference_dir` | `${exp_dir}/${self_name:}` |
| `dataset._target_` | `espnet3.components.data.data_organizer.DataOrganizer` |
| `dataset.recipe_dir` | `${recipe_dir}` inherited from training |
| `dataset.test` | must be filled by the recipe |
| `model._target_` | `espnet2.bin.asr_inference.Speech2Text` |
| `input_key` | `speech` |
| `runner._target_` | `espnet3.systems.esp2_asr.inference.ASRInferenceRunner` |
| `provider._target_` | `espnet3.systems.base.inference_provider.InferenceProvider` |
| `batch_size` | `4`; set `null` for a single-item-only backend |
| `idx_key` | omitted; runtime default `utt_id` |
| `output_keys` | omitted; infer result keys except `idx_key` |
| `parallel` | `env: local`, `n_workers: 1` |
| `contract` | mono 16 kHz `speech` input and `text` output |

`provider`, `runner`, `input_key`, and the standard ASR contract usually need
no recipe overrides. The merged config still needs all of them. Supply the
trained model paths and test datasets:

```yaml
# conf/inference.yaml: override of the ASR TEMPLATE.
dataset:
  test:
    - name: test
      data_src_args:
        split: test

model:
  asr_train_config: ${exp_dir}/config.yaml
  asr_model_file: ${exp_dir}/epoch0_step1_valid.loss.ckpt
  beam_size: 1
```

Here an omitted `data_src` resolves the recipe's `dataset/__init__.py`; an
explicit reference such as `mini_an4/esp2_asr` selects another recipe.
Only `data_src_args` are passed to `Dataset(...)`. Every test entry needs a
unique nonempty `name`, which names its output directory.

## Backend and runner boundary

`InferenceProvider` instantiates `model` and builds the selected test dataset.
`InferenceRunner.forward()` reads the columns named by `input_key`, checks the
contract, calls the backend, formats its result, validates outputs, and
attaches `utt_id` (or `idx_key`). Offline `infer`, packed `InferenceModel`, and
the demo share the configured runner's `forward` path.

The generic runner expects named mappings. A backend that already returns
`{"text": "hello"}` can use the default `InferenceRunner`; an empty subclass
is unnecessary. ASR's native n-best tuples need `ASRInferenceRunner`, which
also converts checked `Audio` values to the waveform representation expected
by `Speech2Text`.

For a backend with a different return structure, override the appropriate hook:

```python
from espnet3.systems.base.inference_runner import InferenceRunner


class CustomRunner(InferenceRunner):
    @staticmethod
    def format_output(result, *, batched):
        def named(value):
            return {"text": value["transcript"]}

        return [named(value) for value in result] if batched else named(result)
```

`prepare_model_inputs(inputs)` can adapt checked inputs
before the call. Keep the default `forward()` to retain shared contract
validation. Configure the class explicitly only when adaptation is needed:

```yaml
runner:
  _target_: src.inference.CustomRunner
```

`output_fn` remains a legacy formatting path and cannot coexist with a
contract. Prefer runner hooks for new contracted recipes. The recipe's code
should use absolute imports.

## Inference contract and input selection

Keep `contract` and `input_key` at the top level of **inference YAML**, not
inside `model`, `runner`, or a dataset class:

```yaml
input_key:
  - speech
  - prompt

contract:
  inputs:
    speech:
      kind: audio
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
```

`input_key` must list exactly the contract input names, including optional
ones. Contract declarations do not replace input selection. A batch must have
consistent optional-input presence. All outputs are required named fields;
undeclared fields other than the identifier are rejected.

Audio inputs with a known rate are resampled by default, or rejected with
`rate_policy: reject`. Bare arrays are assumed to have the declared rate.
Checked audio retains all channels unless an explicit normalizer changes
it; `channels: 1` requires mono and does not itself select a channel. A custom
backend using the generic runner receives an `Audio` object. See the
[contract reference](./contract.html) for formats and custom kinds.

## Batching

`batch_size: null` sends one index per call. A positive integer sends a list
of indices; each backend argument is a list with one value per sample.
A contracted batch returns one mapping per input sample (a list of mappings,
or a supported columnar mapping of equally sized lists). Prefer a list of
mappings when outputs themselves contain lists.

`Speech2Text` for ESPnet2 ASR supports the ASR runner's batch path. Other
backends must implement their own batching; an unsupported or oversized batch
raises rather than automatically retrying one sample at a time. For a
transducer helper, use `espnet2.bin.asr_transducer_inference.Speech2Text`,
`return_decoded_hyp: true`, and `batch_size: null`.

## Output layout and artifacts

```text
${inference_dir}/
  test/
    text.scp
    split.0/
      text.scp
```

SCP rows are `<utt_id> <value_or_path>`. Primitive values are written inline.
`output_keys` optionally restricts persisted result fields; otherwise keys
come from the first checked/formatted result, excluding `idx_key`. The
standard ASR output is `text.scp`, not `hyp.scp` or a copied reference.
References for scoring are read by metrics from `dataset:<column>`.

Non-scalar fields are written below each shard directory and the merged SCP
contains their paths:

```yaml
output_artifacts:
  wav:
    type: wav
    sample_rate: 16000
  posterior:
    type: npy
  details:
    type: json
  custom:
    writer:
      _target_: src.artifacts.write_custom_artifact
```

Built-in types are `wav` (requires `sample_rate`), `npy`, `json`, and `pickle`.
A dict defaults to JSON, a NumPy array or CPU tensor to NPY, and other objects
to pickle. A custom writer returns an existing file path. CUDA tensors and
bare top-level list/tuple output values are rejected by this implementation;
a `segments` field can pass contract validation but still needs an explicit
adapter to a persistable value for the offline writer. A dict can contain
nested lists. Do not confuse a batch's outer list of sample mappings with
one output field's value.

## Parallel execution and resume

```yaml
parallel:
  env: local
  n_workers: 2
  options:
    threads_per_worker: 1
```

The provider builds worker-local state and the runner writes one shard per
worker. `resume` defaults to `true` on `BaseRunner`; a saved manifest must
match the planned indices/batching/shard layout. This baseline does not
fingerprint model or dataset content. Use a fresh output directory or a
clean recomputation after changing inputs or model weights. Recover a stale
lock with `parallel.allow_overwrite_lock: true` only after its writer has
stopped. See [parallel execution](../parallel.html).

## Scoring the outputs

A metrics config binds declared input names to sources:

```yaml
# conf/metrics.yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
```

`hyp: text` reads `text.scp`; `ref: dataset:text` materializes the test set's
transcript column as `<test>/dataset/text.scp`. That cache is reused; delete
it when the reference data changes. The reviewed `measure` method requires
an `infer` stage-config entry, so request `--stages infer measure` and supply
both inference and metrics roles (plus training identity or `--exp_dir`).
