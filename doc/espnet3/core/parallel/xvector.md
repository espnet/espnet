---
title: X-Vector Extraction with Provider and Runner
author:
  name: "Thanapat Trachu"
date: 2026-05-28
---

# X-Vector Extraction with Provider and Runner

This page walks through an example of the Provider / Runner
codes: computing **x-vectors** (speaker embeddings) for every utterance in the LibriTTS VITS recipe.

Read [Provider and Runner](./provider_runner.html) first if you are not familiar with provider and runner yet.

---

## How the pieces fit together

Set the parallel config below and step through each phase to see exactly how
the Provider and Runner interact.

<XVectorAnimation />

---

## Configuration reference

All xvector settings live under the `xvector:` key in `training_config.yaml`.
Select a toolkit to update the config preview, then expand the field reference
for a description of each argument.

<XVectorConfig />

---

## Manifest format

`_load_manifest` is a helper used by `XVectorProvider` in this LibriTTS recipe.
It parses a **tab-separated file** with one utterance per line:

```text
utt_id \t wav_path \t text \t speaker_id
```

Example:

```text
1089-134686-0000	/data/LibriTTS/train-clean-360/1089/134686/1089-134686-0000.wav	HE BEGAN	1089
1089-134686-0001	/data/LibriTTS/train-clean-360/1089/134686/1089-134686-0001.wav	A LONG	1089
```

It returns `utterances` — a `list[(utt_id, wav_path)]` indexed by position —
stored in the env dict so the runner can resolve `utterances[idx]`.

When writing a Provider for a **different task**, replace `_load_manifest` with
whatever data loading your `forward()` needs — a dataset class, a file list, a
HuggingFace dataset, etc. The pattern is the same: load it in the Provider,
store it under a key, declare that key as a parameter in `forward()`.

---

## Output layout

After extraction the output directory looks like:

```text
{save_path}/
  {spk_embed_tag}/
    train/
      {utt_id}.pt          ← one float32 tensor per utterance
    valid/
      ...
    test/
      ...
```

---

## Adapting for a new task

To write your own Provider / Runner for a different task, follow these steps:

**Step 1 — decide what `forward()` needs.**
List everything the per-item function needs: a model, a dataset, a tokenizer,
an output path. Those become the keys of your env dict.

**Step 2 — implement the Provider.**
Return those keys from `build_env_local()`, and mirror the same logic inside
the `setup()` closure in `build_worker_setup_fn()`. Inside `setup()`, use only
the variables captured from the outer scope — `self` is not available because
the closure runs on a remote worker, not on the driver.

**Step 3 — implement the Runner.**
Declare `forward()` as a `@staticmethod` with parameter names that match the
dict keys. Handle both `isinstance(idx, int)` and iterable `idx` if you set
`batch_size`.

**Step 4 — instantiate and call.**
Pass the provider to the runner and call `runner(range(n))`.

Minimal skeleton:

```python
from pathlib import Path
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.base_runner import BaseRunner


class MyProvider(EnvironmentProvider):
    def __init__(self, config, params=None):
        super().__init__(config)
        self.params = params or {}       # runtime args (paths, splits, …)

    def build_env_local(self):
        return {
            "model":   load_my_model(self.config),
            "dataset": load_my_dataset(self.params["data_path"]),
        }

    def build_worker_setup_fn(self):
        config, params = self.config, self.params   # capture, not self
        def setup():
            # self is not available here — use captured config and params
            return {
                "model":   load_my_model(config),
                "dataset": load_my_dataset(params["data_path"]),
            }
        return setup                     # ← function, not dict


class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx, model, dataset, **env):   # names match Provider keys
        if isinstance(idx, int):
            return MyRunner._process_one(idx, model, dataset)
        return [MyRunner._process_one(i, model, dataset) for i in idx]

    @staticmethod
    def _process_one(idx, model, dataset):
        item = dataset[idx]
        result = model(item)
        # write to disk here, or return the result for BaseRunner to collect
        return {"idx": idx, "status": "ok"}


# Usage
provider = MyProvider(config, params={"data_path": "data/manifest.tsv"})
runner   = MyRunner(provider)
runner(range(len(dataset)))
```

---

### Common mistakes

#### **Key name mismatch between Provider and Runner**

If the Provider returns `{"mdl": model}` but `forward()` declares `model`, the
argument is silently not injected and the call raises `TypeError: missing
required argument`.  Always copy-paste key names from the dict directly into
the parameter list.

#### **`build_worker_setup_fn` returns a dict instead of a function**

```python
# ✗ This runs setup on the driver, not the worker — GPU is on the wrong machine
def build_worker_setup_fn(self):
    return {"model": load_model(self.config)}   # wrong

# ✓ This runs setup on each worker when it starts
def build_worker_setup_fn(self):
    config = self.config
    def setup():
        return {"model": load_model(config)}
    return setup
```

#### **Loading the model inside `forward()`**

```python
# ✗ Loads from disk on every single utterance — extremely slow
@staticmethod
def forward(idx, config, **env):
    model = load_model(config)   # wrong — runs N times
    ...

# ✓ Model loaded once in Provider, injected into every forward() call
@staticmethod
def forward(idx, model, **env):  # model comes from Provider env
    ...
```

#### **Not handling both `int` and `list[int]` for `idx`**

When `batch_size` is set, `idx` is a `list[int]`. If `forward()` always
treats `idx` as an `int`, it silently indexes with a list and produces wrong
results or crashes.

```python
@staticmethod
def forward(idx, model, dataset, **env):
    if isinstance(idx, int):
        return process_one(idx, model, dataset)
    return [process_one(i, model, dataset) for i in idx]
```

---

## See also

<DocCards :cols="3">
  <DocCard
    title="Provider and Runner"
    desc="Read the base contracts for EnvironmentProvider and BaseRunner."
    icon="tabler:arrows-split-2"
    href="./provider_runner.html"
  />
  <DocCard
    title="Data Preparation"
    desc="See how providers and runners are used for dataset preparation stages."
    icon="tabler:database"
    href="./data_preparation.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the inference-stage provider pattern."
    icon="tabler:wave-sine"
    href="./inference_provider.html"
  />
  <DocCard
    title="EnvironmentProvider API"
    desc="Read the generated contract for local and worker env setup."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/EnvironmentProvider.html"
  />
  <DocCard
    title="BaseRunner API"
    desc="Read the generated contract for forward, writer hooks, and merge."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/BaseRunner.html"
  />
  <DocCard
    title="Parallel Config"
    desc="Review local, local GPU, and cluster backend settings."
    icon="tabler:settings"
    href="./provider_runner.html"
  />
</DocCards>
