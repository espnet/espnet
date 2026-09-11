---
title: ESPnet3 Publication Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Publication Configuration

This page describes the current `publication.yaml` used by:

```bash
python run.py \
  --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

`pack_model` and `upload_model` are implemented by
[`espnet3.utils.publication_utils`](https://github.com/espnet/espnet/blob/master/espnet3/utils/publication_utils.py).
There is no ESPnet2-vs-ESPnet3 packing strategy switch — `pack_model` always
copies `training_config.exp_dir` plus a few configured paths into one bundle.

## Minimum required keys

Typical publication runs need:

- `training_config` (for `recipe_dir` / `exp_dir`)
- `publication_config.upload_model.hf_repo`, for `upload_model`

Optional but useful:

- `inference_config`, `metrics_config` — bundled into `conf/` and used to find
  `metrics.json` for the README results table
- `pack_model.include`, `pack_model.exclude` — extra copy/exclusion control
- `pack_model.files`, `pack_model.yaml_files` — explicit named artifacts

## Config sections overview

| Section | Description |
| --- | --- |
| `exp_tag`, `exp_dir`, `inference_dir`, `data_dir` | path scaffold, usually inherited from `training_config`/`inference_config` |
| `pack_model.out_dir` | output bundle directory |
| `pack_model.allow_overwrite` | allow reusing an existing `out_dir` |
| `pack_model.include`, `exclude` | extra copy and exclusion control (applied only to the bulk copy step) |
| `pack_model.files`, `yaml_files` | explicit named artifacts copied into the bundle and registered in `meta.yaml` |
| `pack_model.readme`, `readme_context` | README template and extra template values |
| `pack_model.include_model_detail` | include `repr(model)` in the README |
| `upload_model` | Hugging Face model-repo upload settings |

## Default values

| Key | Default value |
| --- | --- |
| `exp_tag` | `${self_name:}` |
| `exp_dir` | `exp/${exp_tag}` |
| `inference_dir` | `${exp_dir}/inference` |
| `pack_model.out_dir` | `${exp_dir}/model_pack` |
| `pack_model.allow_overwrite` | `false` |
| `pack_model.include_model_detail` | `false` |
| `pack_model.readme` | `${config_path:../src/hf_model_readme.md}` |
| `pack_model.include` | `[src, dataset]` in TEMPLATE |
| `pack_model.exclude` | `[inference, inference_*, last.ckpt, "**/step*.ckpt", "**/*.log", "**/tensorboard/**", "**/wandb/**"]` |
| `upload_model.private` | `false` |
| `upload_model.update` | `false` |
| `upload_model.delete_patterns` | `["*"]` (only used when `update: true`) |

## Typical usage

### Pack and upload in one run

```bash
python run.py \
  --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

`run.py` propagates `training_config.exp_tag`/`exp_dir` into `publication_config`,
and `inference_config.inference_dir` into `publication_config.inference_dir`
before the stage runs, so run `infer` and `measure` first if you want the
bundle to include evaluation results.

### Upload only

If the bundle already exists, `upload_model` can run alone as long as
`pack_model.out_dir` (or the derived default) points at it:

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

```bash
python run.py \
  --stages upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

## Minimal example

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

This is the smallest user override; `pack_model` uses its TEMPLATE defaults.

## `pack_model`

`pack_model` builds the bundle in this order:

1. **Bulk copy** — copy `${exp_dir}`, then each path in `include`, into the
   bundle. `exclude` patterns are applied only during this step.
2. **Named artifacts (`files`)** — copy each entry individually and register
   it in `meta.yaml`. `exclude` does not apply here.
3. **Named YAML artifacts (`yaml_files`)** — same as `files`, but paths inside
   the YAML are rewritten to bundle-relative `${recipe_dir}/...` form.
4. **Configs** — `training_config`/`inference_config`/`metrics_config`/
   `publication_config` are written into `conf/` with paths rewritten the same
   way.
5. `meta.yaml` is written at the bundle root.

### `out_dir` and `allow_overwrite`

`out_dir` is the final bundle directory (default `${exp_dir}/model_pack`). If
it already exists, `pack_model` raises unless `allow_overwrite: true`, in
which case the existing directory is removed first.

Typical bundle layout:

```text
${exp_dir}/model_pack/
  conf/
    training.yaml
    inference.yaml   # if --inference_config was passed
    metrics.yaml      # if --metrics_config was passed
    publication.yaml
  src/
  dataset/
  <copy of exp_dir>
  files/
  yaml_files/
  meta.yaml
  README.md
  metrics.json        # if found under inference_dir
```

`conf/*.yaml` only includes the configs actually passed to `run.py`.
`metrics.json` is copied from `inference_dir` (checked in order:
`publication_config.inference_dir`, then `metrics_config.inference_dir`, then
`inference_config.inference_dir`) when it exists.

### `include` and `exclude`

| Key | Description |
| --- | --- |
| `include` | extra paths (globs supported) copied alongside `exp_dir` during the bulk copy |
| `exclude` | glob patterns skipped during the bulk copy only |

TEMPLATE example:

```yaml
pack_model:
  include:
    - src
    - dataset
  exclude:
    - inference
    - inference_*
    - last.ckpt
    - "**/step*.ckpt"
    - "**/*.log"
    - "**/tensorboard/**"
    - "**/wandb/**"
```

A real recipe example (`egs3/mini_an4/asr/conf/publication.yaml`):

```yaml
pack_model:
  include:
    - ${recipe_dir}/src
    - ${recipe_dir}/dataset
    - ${data_dir}/**/bpe.model
    - ${data_dir}/**/bpe.vocab
    - ${data_dir}/**/tokens.txt
```

### `files` and `yaml_files`

These keys copy explicit artifacts into the bundle and register them under
`meta.yaml`'s `files`/`yaml_files` maps. Both apply regardless of `exclude`.
`yaml_files` entries additionally get any recipe paths inside them rewritten
to bundle-relative form.

```yaml
pack_model:
  files:
    bpemodel: ${data_dir}/bpe_5000/bpe.model
  yaml_files:
    tokenizer_config: ${data_dir}/bpe_5000/config.yaml
```

### `readme` and `readme_context`

`pack_model` renders `README.md` from the template at `readme`
(default `${config_path:../src/hf_model_readme.md}`, i.e. the recipe's
`src/hf_model_readme.md`). `readme_context` values override the context
`pack_model` infers automatically (repo name, recipe name, system name, model
summary, and a results table built from `metrics.json` when present). Set
`include_model_detail: true` to also embed `repr(model)`.

## `meta.yaml`

`pack_model` writes `meta.yaml` at the bundle root with `schema_version`,
the `files`/`yaml_files` maps (as bundle-relative paths), and environment
versions (`torch`, `espnet`, `python`). `InferenceModel.from_packed()` reads
`meta.yaml.yaml_files.inference_config` to locate the packed inference config.

## `InferenceModel`

Published bundles are consumed through:

- `espnet3.publication.InferenceModel`

```python
import soundfile as sf

from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained(
    "espnet/your-model-tag",
    trust_user_code=True,
)

audio, sample_rate = sf.read("sample.wav", dtype="float32")
result = model(audio)
print(result)
```

- `InferenceModel.from_packed(pack_dir, ...)` loads a local bundle directory
  directly.
- `InferenceModel.from_pretrained(model_tag, ...)` downloads a Hugging Face
  Hub bundle first, then delegates to `from_packed`.
- Pass `trust_user_code=True` when the packed `inference.yaml` references
  bundled `src.*` modules; the bundle root is then added to `sys.path`.

See [Publication stages](../stages/publish.html) for stage flow.

## `upload_model`

Minimal example:

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

| Key | Description |
| --- | --- |
| `hf_repo` | required; full repo id, e.g. `yourname/your-model-repo` |
| `private` | create the repo as private (default `false`) |
| `update` | allow uploading over an existing repo (default `false`) |
| `delete_patterns` | glob patterns of existing repo files to delete first when `update: true` (default `["*"]`) |

Current behavior:

- uploads `pack_model.out_dir` (or its default) to `hf_repo` via
  `huggingface_hub`
- raises if that directory does not exist, or if the repo already exists and
  `update` is not `true`
- requires a Hugging Face token from `hf auth login`

## Notes

- use `publication.yaml`, not `publish.yaml`
- use `--publication_config`, not `--publish_config`
- `pack_model` and `upload_model` are stage config blocks inside `publication.yaml`

## Related pages

- [Publication stages](../stages/publish.html)
- [Inference configuration](./infer_config.html)
- [Inference stage](../stages/inference.html)
- [Demo configuration](./demo_config.html)
