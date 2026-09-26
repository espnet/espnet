---
title: ESPnet3 Inference Stage
author:
  - name: "Masao Someki"
  - name: "Elias Naske"
date: 2026-05-21
---

# ESPnet3 Inference Stage

The `infer` stage runs model inference on the provided test set(s) and writes the outputs to disk.
The resulting files are used to measure model performance in the [`measure`](./metrics.html) stage.

## 1. Run

```bash
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml
```

`--inference_config` is required for `infer`. `--training_config` is also
required unless `inference_config` defines its own `exp_tag` or a concrete
`exp_dir` -- otherwise `run.py` cannot resolve `exp_dir`/`inference_dir` and
raises a `ValueError` before running the stage. The common pattern is to pass
both flags, so `training_config.exp_tag`/`exp_dir` are copied into
`inference_config` automatically.

## 2. Configuration

Keep the core settings in `inference.yaml`. For the full list, see
[Inference configuration](../config/infer_config.html).

| Config section  | Description                                   |
| --------------- | --------------------------------------------- |
| `model`         | model to run inference with                   |
| `dataset`       | definition of the test set(s); each entry under `dataset.test` needs a `name`, which becomes its output subdirectory |
| `inference_dir` | root output location (default `${exp_dir}/${self_name:}`, i.e. `${exp_dir}/<inference config filename>`) |
| `input_key`     | dataset field or fields passed into the model |
| `output_fn`     | function used to format the output files      |
| `parallel`      | local or distributed runner settings, see [below](#multi-gpu-and-parallel-inference) |
| `runner.resume` | whether to reuse a previous run's shard outputs, see [below](#resuming-a-partial-or-failed-run) |

`egs3/TEMPLATE/asr/conf/inference.yaml` ships `dataset.test:` and `model:` blank as placeholders --
both must be filled in before running `infer`. Leaving `dataset.test` unset fails immediately with a
bare `TypeError: 'NoneType' object is not iterable`; leaving `model` unset fails later, inside a
worker process, with `TypeError: 'NoneType' object is not callable`. Neither error names the missing
config field, so check `dataset.test`/`model` first if `infer` fails this way.

## 3. Outputs

Inference writes one directory per test set:

```text
<inference_dir>/
└── <test_name>/
    ├── hyp.scp
    └── ...
```

The filenames are determined by:

- `output_keys` when it is set
- otherwise the keys returned by `output_fn` for the first sample, excluding
  `idx_key`

### SCP Files

Within a `.scp` file, each line represents an utterance and takes the following form:

```text
{utt_id} {value}
```

The value is determined by `output_fn` and can be either:
- a scalar value (`str`, `int`, `float`, `bool`)
- a non-scalar value (e.g. `dict`, `numpy.ndarray`, `torch.tensor`)

Scalar values are written directly into SCP files.
Non-scalar values are written as artifacts to a file, and the SCP stores a path to said file.
Artifacts are written under:
```text
<inference_dir>/<test_name>/<field_name>/
```
The file type depends on the return type of `output_fn`:

| Value type          | Default artifact type | Saved as |
| ------------------- | --------------------- | -------- |
| `dict`              | `json`                | `.json`  |
| `numpy.ndarray`     | `npy`                 | `.npy`   |
| CPU `torch.Tensor`  | `npy`                 | `.npy`   |
| other Python object | `pickle`              | `.pkl`   |

The config can also be set to force other types, such as `wav`.
E.g., if `output_fn` returns:

```python
{
    "utt_id": "utt1",
    "audio": wav_numpy,
}
```

and `inference.yaml` contains:

```yaml
output_artifacts:
  audio:
    type: wav
    sample_rate: 16000
```

then inference writes:

```text
<inference_dir>/
└── <test_name>/
    ├── audio.scp
    └── audio/
        ├── utt1.wav
        └── utt2.wav
```

and `audio.scp` stores the generated `.wav` paths.


### Custom artifact writers

If you want to save a custom type such as PNG, add a writer function and point
to it from config.

Example config:

```yaml
output_artifacts:
  image:
    writer:
      _target_: src.inference.write_png_artifact
```

Example function:

```python
from pathlib import Path


def write_png_artifact(*, value, output_path):
    path = Path(output_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    value.save(path)
    return path
```

The writer must return the written path. That path is stored in the SCP file.

## 4. Resuming a partial or failed run

Each test set's work is split into shards under
`<inference_dir>/<test_name>/split.<shard_id>/`. A shard directory containing
a `.done` marker is treated as complete. By default (`runner.resume: true`,
the `InferenceRunner`/`BaseRunner` default), re-running `infer` with the same
`--inference_config` skips every shard that is already marked done and only
computes the remaining ones -- this is what makes it safe to re-run `infer`
after an interrupted or partially failed job.

::: warning
Resume only checks that the **shard plan** (how many shards, and which
dataset indices went into each one) is unchanged between runs -- it does not
fingerprint the model or provider config. If you change
`inference_config.model` (e.g. point at a different checkpoint), `input_key`,
`output_fn`, or any other provider setting and re-run with `resume: true` (the
default), already-`.done` shards are skipped and their stale SCP output from
the *previous* model/config is reused silently.

To force a full re-run after changing the model or provider config, either:

- pass `runner: {resume: false}` in `inference.yaml` for that run, or
- delete `<inference_dir>/<test_name>/` (or the whole `<inference_dir>`)
  before re-running.
:::

## 5. Multi-GPU and parallel inference

`inference_config.parallel` controls sharding:

```yaml
parallel:
  env: local
  n_workers: 1
```

::: warning
`n_workers` is only read when `parallel.env` is **not** `"local"`. With the
default `env: local`, exactly one shard is created and it always runs
sequentially on the driver process, regardless of `n_workers` -- raising
`n_workers` under `env: local` has no effect and produces no warning. To
actually parallelize inference across workers, set `parallel.env` to a
non-`local` backend (see [Scaling: Inference](../guides/scaling/inference.html))
together with `n_workers`.
:::

Running `infer` under multi-GPU DDP training in the same invocation is also
affected by the rank-guard gap described in [the train stage
docs](./train.html): running `--stages train infer` (or the full default
stage list) under `trainer.devices > 1` with the local subprocess launcher
causes every DDP rank to re-run `infer` against the same `inference_dir`
concurrently, which fails with a shard-lock error or races on shard output.
Run `infer` as its own single-process `python run.py --stages infer ...`
invocation after training completes.

## 6. Implementation Details
### Inference Providers and Runners

Inference is implemented as a Provider/Runner loop.

The provider is responsible for:

- building the dataset for the active test set
- instantiating the model
- exposing config-derived runtime parameters

The runner is responsible for:

- pulling one sample or one batch from the dataset
- calling the model with the configured `input_key`
- normalizing the result through `output_fn`
- returning values that can be written into SCP files

Conceptually:
```python
provider = InferenceProvider(config)
runner = InferenceRunner(provider=provider, async_mode=False)
results = runner(range(len(provider.build_dataset(config))))
```

### Batch Inference

Inference can be run batched by setting the top-level `batch_size` in
`inference.yaml` (it is not nested under `runner:`).

For example:
```yaml
batch_size: 4
```

This will pass a list of indices to `InferenceRunner.forward()`.

::: warning `batch_size: 1` is not the same as leaving it unset
Any integer `batch_size`, including `1`, chunks indices into lists and calls `forward()` on the
*batched* code path (`is_batched=True` in
[`InferenceRunner.forward`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/inference_runner.py)) --
`model`/`output_fn` then receive list-wrapped inputs even for a "batch" of one. If your model or
`output_fn` only supports single-sample calls, leave `batch_size` unset (or `null`) rather than
setting it to `1`; a mismatch here fails with `RuntimeError: Batched inference failed...`.
:::

### `output_fn`

`output_fn` is called right after the model returns.

If provided, `output_fn` is called as:

```python
output_fn(data=data, model_output=model_output, idx=idx)
```

It should return a dict for a single sample, or a list of dicts for batched
inference.

Typical output:

```python
{
    "utt_id": "utt1",
    "hyp": "hello world",
}
```

The base runner accepts either a single index or a list of indices. That is why
`output_fn` must be able to handle:

- a single sample plus scalar `idx`
- or batched input where `data` is a list and `idx` is a list

Minimal single-sample example:

```python
def build_output(*, data, model_output, idx):
    return {
        "utt_id": data["uttid"],
        "hyp": model_output["text"],
        "ref": data.get("text", ""),
    }
```


## 7. Using a custom model

There a two common paths when using a custom models:

1. keep `InferenceRunner` and replace only `model` and `output_fn`
2. replace `InferenceRunner` when the normal flow is not enough

It is generally recommended to keep `InferenceRunner` and only replace it for special use cases.

### Example: Custom Decoding Algorithm

This is the common case:

- you want to keep the same dataset
- you want to keep the same SCP writing path
- but you want your own decoding algorithm

In that case, keep the default runner and replace only `model` and `output_fn`.

Example `inference.yaml`:

```yaml
dataset:
  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test

model:
  _target_: src.inference.MyGreedyDecoder
  checkpoint_path: ${exp_dir}/last.ckpt
  beam_size: 1

input_key: speech
output_fn: src.inference.build_output

provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider
runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
```

Example `src/inference.py`:

```python
from pathlib import Path

import torch


class MyGreedyDecoder:
    def __init__(self, checkpoint_path, beam_size=1):
        self.checkpoint_path = Path(checkpoint_path)
        self.beam_size = beam_size
        self.model = self._load_model()

    def _load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model = checkpoint["model"]
        model.eval()
        return model

    def __call__(self, speech):
        tokens = self.model.decode(speech, beam_size=self.beam_size)
        text = self.model.tokenizer.decode(tokens)
        return {"text": text, "tokens": tokens}


def build_output(*, data, model_output, idx):
    return {
        "utt_id": data.get("uttid", str(idx)),
        "hyp": model_output["text"],
        "token_ids": " ".join(str(v) for v in model_output["tokens"]),
        "ref": data.get("text", ""),
    }
```

The runtime order is:

1. `InferenceRunner` loads one sample from the dataset
2. it calls `model(**inputs)`
3. it calls `build_output(...)`
4. it writes `hyp.scp`, `token_ids.scp`, and `ref.scp`

### When to replace `InferenceRunner`

Replace the runner only when `model -> output_fn -> SCP` is not enough.

Examples:

- streaming decode with internal state
- multi-step search with custom batching
- non-standard output validation

Minimal custom runner example:

```python
from espnet3.systems.base.inference_runner import InferenceRunner


class MyInferenceRunner(InferenceRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        model_output = model.decode_stream(data["speech"])
        return {
            "utt_id": data.get("uttid", str(idx)),
            "hyp": model_output["text"],
            "ref": data.get("text", ""),
        }
```

Config:

```yaml
runner:
  _target_: src.inference.MyInferenceRunner
```

Use this path only when `output_fn` is not enough.

## Related pages

**Stage API:** [`infer`](../../guide/espnet3/systems/infer.html),
[`InferenceProvider`](../../guide/espnet3/systems/InferenceProvider.html),
[`InferenceRunner`](../../guide/espnet3/systems/InferenceRunner.html), and
[`write_artifact`](../../guide/espnet3/utils/write_artifact.html). Run this after
[training](./train.html) and before [measurement](./metrics.html).

<DocCards :cols="3">
  <DocCard
    title="Inference configuration"
    desc="See all options for cofiguring the inference stage."
    icon="tabler:file-code"
    href="../config/infer_config.html"
  />
  <DocCard
    title="Measure stage"
    desc="Information on the measure stage."
    icon="tabler:puzzle"
    href="./metrics.html"
  />
  <DocCard
    title="Provider / Runner"
    desc="Learn how providers and runners work together during inference."
    icon="tabler:tool"
    href="../core/parallel/provider_runner.html"
  />
</DocCards>
