# Inference stage

`infer` reads `dataset.test`, calls the configured backend through a runner, and
writes one SCP per named model output. The same runner `forward` path is reused
by packed `InferenceModel` calls and demos.

## Run

From an ASR recipe such as `egs3/mini_an4/esp2_asr`:

```bash
python run.py --stages infer \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml
```

The training role supplies experiment identity and other inherited keys. An
existing experiment can supply saved configuration via `--exp_dir` instead.
An inference config with unresolved `${exp_dir}` cannot provide identity by
itself. The role defaults come from `egs3.TEMPLATE.esp2_asr`; a recipe overlay
normally specifies only values that differ.

Every `dataset.test` entry needs a unique non-empty `name`, which becomes its
output directory. The TEMPLATE's test list is a placeholder and must be filled.
`input_key` is still required with a contract: it selects dataset columns and
must match `contract.inputs`.

## Contracts and the backend boundary

An ordinary ASR recipe inherits these settings from the ASR TEMPLATE:

```yaml
input_key: speech

model:
  _target_: espnet2.bin.asr_inference.Speech2Text

runner:
  _target_: espnet3.systems.esp2_asr.inference.ASRInferenceRunner

contract:
  inputs:
    speech:
      kind: audio
      sample_rate: 16000
  outputs:
    text:
      kind: text
```

Supply trained model/config paths in the recipe's `model` block. The shared
provider supplies construction-time device settings. The contract converts
rate-bearing audio inputs to `Audio`, resampling by default; a bare array carries
no source rate to verify. Channel count is unconstrained unless explicitly set;
channel selection or averaging requires an explicit normalizer. `ASRInferenceRunner`
unwraps `Audio` to the backend's samples-first array and converts the top n-best
hypothesis to `{"text": ...}` before output validation. Empty text remains valid.

For a backend that already accepts checked inputs and returns named outputs,
use the default `InferenceRunner` directly. A minimal complete example in the
recipe's `src/inference.py` is:

```python
from espnet3.contract.kinds.audio import Audio


class DurationBackend:
    def __call__(self, speech):
        def one(audio):
            if not isinstance(audio, Audio):
                raise TypeError("expected contract-checked Audio")
            return {"duration": audio.duration}

        return [one(audio) for audio in speech] if isinstance(speech, list) else one(speech)
```

Configure it in `conf/inference.yaml` (retain the recipe's test dataset):

```yaml
model:
  _target_: src.inference.DurationBackend
runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
input_key: speech
contract:
  inputs:
    speech:
      kind: audio
      sample_rate: 16000
  outputs:
    duration:
      kind: number
```

An unrelated backend must not inherit the ASR runner's n-best formatter. Keep the
default runner when the backend already fits its protocol. Override
`prepare_model_inputs` to adapt checked inputs or `format_output(result, *, batched)`
to adapt backend results when needed; a custom `forward` implementation owns its
own contract checks. A pass-only subclass adds no behavior.

`contract` and legacy `output_fn` cannot be combined. Without a contract,
`output_fn(data=..., model_output=..., idx=...)` may return one mapping or a list
of mappings for a batch. New recipes should return named outputs or adapt them
with runner hooks instead. The runner attaches `utt_id` (or `idx_key`) from the
sample identifier, falling back to the dataset index.

## Batching and optional inputs

`batch_size` belongs at the top level of `inference.yaml`:

```yaml
batch_size: 4
```

Any integer, including `1`, selects batch calls: each input keyword contains a
list. Set `batch_size: null` for a single-item-only backend. The default runner
does not silently retry unsupported batches. The ESPnet2 ASR backend supports
batched decoding; the transducer TEMPLATE recommends single calls and its own
backend settings. Within one batch, an optional input must be present for every
item or absent for every item.

## SCP files and artifacts

For the ASR contract, the final output is `text.scp`, not a copied reference:

```text
<inference_dir>/<test_name>/
├── text.scp
└── split.0/
    ├── text.scp
    └── done
```

Each SCP row is `<utt_id> <value-or-path>`. Primitive scalar values are written
inline. Non-scalar values are stored under the worker shard directory:

```text
<inference_dir>/<test_name>/split.<id>/<field>/<utt_id>.<extension>
```

Merged SCPs at the test-set root point to those shard-local artifacts. Arrays and
CPU tensors default to NPY, mappings to JSON, and other supported objects to
pickle; a top-level list/tuple field is rejected rather than used as an arbitrary
artifact. CUDA tensors must be moved to CPU. Configure WAV serialization explicitly:

```yaml
output_artifacts:
  wav:
    type: wav
    sample_rate: 24000
```

`output_keys` optionally filters persisted named fields; otherwise keys are
inferred from the first result, excluding the identifier. It does not replace an
output contract. For custom serialization:

```yaml
output_artifacts:
  image:
    writer:
      _target_: src.inference.write_png_artifact
```

```python
from pathlib import Path


def write_png_artifact(*, value, output_path):
    path = Path(output_path).with_name(Path(output_path).name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    value.save(path)
    return path
```

## Parallel execution and resume

`parallel.env: local` executes sequentially on the driver; increasing `n_workers`
there alone does not create workers. Use a configured non-local backend for
Dask-distributed shards; see [parallel execution](../parallel.html).

The current `BaseRunner` resumes from completed `done` markers and validates the
shard plan, not model or dataset identity. After changing the checkpoint, inputs,
contract, or provider settings, use a fresh output directory or force recomputation:

```yaml
runner:
  resume: false
```

For a stale shard lock, `parallel.allow_overwrite_lock: true` is available only
when the prior writer has stopped. Non-training stage bodies are not rank-gated
by `run_stages`; launch inference as a separate single-process command after a
multi-GPU training job.

## Source and next steps

- [Inference entry point](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/inference.py)
- [Shared runner and hooks](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/inference_runner.py)
- [ASR adapter](https://github.com/espnet/espnet/blob/master/espnet3/systems/esp2_asr/inference.py)
- [Inference configuration](../config/infer_config.html)
- [Measure model outputs](./metrics.html)
