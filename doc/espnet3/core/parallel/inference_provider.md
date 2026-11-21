---
title: ESPnet3 Inference Provider
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Inference Provider

[`InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html) is
the convenience provider for inference-time environment construction.

ESPnet3 currently has two related layers:

- [`espnet3.parallel.inference_provider.InferenceProvider`](../../../guide/espnet3/parallel/InferenceProvider.html)
- [`espnet3.systems.base.inference_provider.InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html)

The stage-facing class used by `inference.yaml` is the second one:

- [`espnet3.systems.base.inference_provider.InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html)

That class is the main focus of this page.

It is a small provider with a clear job:

- build the dataset for one test set
- build the model
- return them as a plain env dictionary for the runner

This is the provider used by the base inference stage.

## What it returns

The provider returns an env like this:

```python
{
    "dataset": dataset,
    "model": model,
    **params,
}
```

`dataset` and `model` are the core entries.

Any extra `params` are merged into the env too.
That makes it easy to inject small runtime values without changing the runner.

## Where it fits

The flow is:

1. `infer()` resolves one `test_set`
2. [`InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html)
   builds the dataset and model for that set
3. [`InferenceRunner.forward(...)`](../../../guide/espnet3/systems/InferenceRunner.html)
   receives the env by keyword name
4. outputs are materialized under `inference_dir`

So this provider is the bridge between:

- `inference.yaml`
- the instantiated dataset/model objects
- the runner that executes inference over sample indices

## Base behavior

The base class already implements:

- [`build_env_local()`](../../../guide/espnet3/systems/InferenceProvider.html)
- [`build_worker_setup_fn()`](../../../guide/espnet3/systems/InferenceProvider.html)
- [`build_dataset(config)`](../../../guide/espnet3/systems/InferenceProvider.html)
- [`build_model(config)`](../../../guide/espnet3/systems/InferenceProvider.html)

In many cases, you only need to subclass it if your dataset or model setup is
special.

**build_dataset(config)**

See the
[`build_dataset` API docs](../../../guide/espnet3/systems/InferenceProvider.html)
for the exact contract.

The default implementation:

1. instantiates `config.dataset`
2. expects that object to be a `DataOrganizer`
3. reads `config.test_set`
4. returns `organizer.test[test_set]`

That means the provider is designed for one concrete test split at a time.

```python
dataset = instantiate(config.dataset)
return dataset.test[config.test_set]
```

**build_model(config)**

See the
[`build_model` API docs](../../../guide/espnet3/systems/InferenceProvider.html)
for the exact contract.

The default implementation:

1. resolves the visible device
2. instantiates `config.model`
3. passes `device=...` into the model constructor

The device resolution logic checks:

1. `config.device`
2. `config.device_index`
3. `config.local_rank`
4. `LOCAL_RANK`
5. falls back to `cuda:0` or `cpu`

This is important for `local_gpu` or worker-per-GPU execution.

## Example: use the default behavior

If your `inference.yaml` already defines:

- `dataset`
- `model`
- `provider`
- `runner`

then you can usually point `provider` at the base provider directly.

```yaml
provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider

runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test:
    - name: test_clean
      data_src: egs3.mini_an4.asr.dataset.builder
      data_src_args:
        split: test

model:
  _target_: egs3.mini_an4.asr.src.inference.SimpleInferenceModel
```

In that setup:

- `infer()` sets `config.test_set = "test_clean"`
- the provider returns `organizer.test["test_clean"]`
- the model is instantiated once per process

## Example: override dataset construction

Override `build_dataset()` when:

- your test dataset is not under `organizer.test`
- you need extra filtering
- you need a wrapper dataset for inference-only normalization

```python
from hydra.utils import instantiate

from espnet3.systems.base.inference_provider import InferenceProvider


class MyInferenceProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        organizer = instantiate(config.dataset)
        dataset = organizer.test[config.test_set]
        return MyDatasetWrapper(dataset, normalize_text=True)
```

This keeps the rest of the provider behavior unchanged.

## Example: override model construction

Override `build_model()` when:

- the model needs custom preload logic
- you need to load auxiliary artifacts
- you want to attach runtime helpers before inference starts

```python
from hydra.utils import instantiate

from espnet3.systems.base.inference_provider import InferenceProvider


class MyInferenceProvider(InferenceProvider):
    @staticmethod
    def build_model(config):
        model = instantiate(config.model, device="cpu")
        model.load_extra_assets(config.extra_assets)
        model.eval()
        return model
```

If you override this method, keep device handling explicit.
Do not assume physical GPU ids from `CUDA_VISIBLE_DEVICES`.

## Example: use params for small runtime values

`params` are merged into the env after dataset/model construction.

That is useful for small values that the runner needs by name.

```python
provider = InferenceProvider(
    inference_config=cfg,
    params={
        "beam_size": 8,
        "return_attention": False,
    },
)
```

Then the runner can receive them directly:

```python
@staticmethod
def forward(idx, dataset, model, beam_size, return_attention, **env):
    sample = dataset[idx]
    return model(sample, beam_size=beam_size, return_attention=return_attention)
```

Use `params` for lightweight runtime flags.
Do not use them to smuggle large driver-side objects into workers.

## How output fields are actually chosen

The [`infer()`](../../stages/inference.html) entrypoint puts `idx_key` and
`output_keys` into the **provider's** `params` (so they arrive in the worker
env dict), and separately passes `idx_key`/`hyp_key`/`ref_key` to the
[`InferenceRunner`](../../../guide/espnet3/systems/InferenceRunner.html)
constructor. Only the provider-supplied env values reach
`InferenceRunner.write_record(...)` (it receives `**env`, not `self`), so in
the shipped pipeline it is `output_keys` — not the runner's own `hyp_key`/
`ref_key` attributes — that determines which `<field>.scp` files get written.
If you instantiate `InferenceRunner` directly instead of going through
`infer()`, make sure `hyp_key`/`ref_key` (or `output_keys`) are also part of
what your provider merges into the env, or they will not affect validation or
output at all.

## Local vs worker behavior

In local mode:

- [`build_env_local()`](../../../guide/espnet3/systems/InferenceProvider.html)
  runs once on the driver

In Dask mode:

- [`build_worker_setup_fn()`](../../../guide/espnet3/systems/InferenceProvider.html)
  returns a zero-argument setup function
- that setup function runs once per worker
- each worker builds its own dataset/model pair

This is why the base implementation avoids capturing `self` in the worker
setup closure.

## When to subclass

Use the base class as-is when:

- `config.dataset` is a normal `DataOrganizer`
- `config.model` is directly instantiable
- one `test_set` maps cleanly to one dataset

Subclass it when:

- dataset selection is custom
- model loading needs extra steps
- worker-local setup needs extra artifacts

## Common mistakes

- Forgetting that `build_dataset()` expects `config.test_set`
- Returning the full organizer instead of one dataset split
- Putting large objects into `params`
- Hard-coding a physical GPU id in `build_model()`
- Reimplementing `build_env_local()` when overriding `build_dataset()` or
  `build_model()` would be enough

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Provider / Runner"
    desc="Read the base contract for providers, runners, and env injection."
    icon="tabler:arrows-split-2"
    href="./provider_runner.html"
  />
  <DocCard
    title="Parallel Config"
    desc="Configure `local`, `local_gpu`, SSH, or HPC backends."
    icon="tabler:settings-2"
    href="./provider_runner.html"
  />
  <DocCard
    title="Inference Config"
    desc="See how `provider`, `runner`, `dataset`, and `model` are written in YAML."
    icon="tabler:wave-sine"
    href="../../config/infer_config.html"
  />
  <DocCard
    title="Inference Stage"
    desc="See how the provider is used inside the stage entrypoint."
    icon="tabler:player-play"
    href="../../stages/inference.html"
  />
  <DocCard
    title="Systems InferenceProvider API"
    desc="Read the generated docstring contract for the stage-facing provider."
    icon="tabler:book"
    href="../../../guide/espnet3/systems/InferenceProvider.html"
  />
  <DocCard
    title="InferenceRunner API"
    desc="Read how env keys are consumed by the base inference runner."
    icon="tabler:file-code"
    href="../../../guide/espnet3/systems/InferenceRunner.html"
  />
  <DocCard
    title="Parallel InferenceProvider API"
    desc="Compare the lower-level parallel provider base class."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/InferenceProvider.html"
  />
  <DocCard
    title="X-Vector Extraction"
    desc="Concrete provider/runner example outside inference: speaker embedding extraction."
    icon="tabler:file-code"
    href="./xvector.html"
  />
</DocCards>
