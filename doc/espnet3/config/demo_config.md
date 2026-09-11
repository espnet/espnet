---
title: ESPnet3 Demo Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Demo Configuration

This page explains the `demo.yaml` schema used by `pack_demo`/`upload_demo`.
It ties a Gradio demo to a **packed model** (`model.dir_or_tag`), not directly
to an `inference.yaml` path. For a deeper dive on demo behavior, see the
[Demo guide](../stages/demo.html).

```bash
python run.py --stages pack_demo   --demo_config conf/demo.yaml
python run.py --stages upload_demo --demo_config conf/demo.yaml
```

## Minimum required keys

Required:

- `model.dir_or_tag`
- `ui.app_script`

Common optional:

- `ui.title`, `ui.description`, `ui.inputs`, `ui.outputs`
- `pack.include`, `pack.exclude`, `pack.requirements`, `pack.readme`
- `upload_demo` (only for uploads)

`model.dir_or_tag` is usually left to be filled in automatically: when both
`--training_config` and (optionally) `--publication_config` are passed to
`run.py`, it is inserted from `publication_config.upload_model.hf_repo` (if
set) or `publication_config.pack_model.out_dir`, falling back to
`training_config.exp_tag`/`exp_dir` via the TEMPLATE default
`exp/${exp_tag}/model_pack`. Running `--demo_config` alone, with no
`--training_config`/`--publication_config` to supply `exp_tag`, fails to
resolve this default — set `model.dir_or_tag` explicitly in that case.

## Config sections overview

| Section | Description |
| --- | --- |
| `model.dir_or_tag` | local packed-model directory, or a Hugging Face tag understood by `espnet_model_zoo` |
| `model.trust_user_code` | forwarded to `InferenceModel` when loading the packed model |
| `model.call_args` | constant kwargs passed at call time: `model(sample, **call_args)` |
| `ui.app_script` | Gradio launcher script copied into the bundle as `app.py` |
| `ui.title`, `ui.description` | demo title and description (a file path is copied in; inline text is shown as-is) |
| `ui.inputs`, `ui.outputs` | positional Gradio component specs (`key`, `type`, `label`) |
| `pack.out_dir` | packed demo output directory |
| `pack.include`, `exclude` | extra files to bundle / exclude |
| `pack.requirements` | Python packages written to `requirements.txt` |
| `pack.readme`, `readme_context` | Hugging Face Space `README.md` template and context |
| `upload_demo` | Hugging Face Space upload settings |

## Default values

| Key | Default value |
| --- | --- |
| `recipe_dir` | `.` |
| `model.dir_or_tag` | `exp/${exp_tag}/model_pack` |
| `model.trust_user_code` | `false` |
| `model.call_args` | `{}` |
| `ui.app_script` | `src/app.py` |
| `ui.title` | `${set_corpus_and_system:} demo` |
| `ui.description` | `README.md` |
| `pack.out_dir` | `demo` |
| `pack.include` / `exclude` | `[]` |
| `pack.requirements` | `[git+https://github.com/espnet/espnet.git]` |
| `pack.readme` | `${config_path:../src/hf_demo_readme.md}` |
| `upload_demo.repo_type` | `space` (default in `upload_demo()`) |
| `upload_demo.update` | `false` |
| `upload_demo.delete_patterns` | `["*"]` (only used when `update: true`) |

## Core config layout

```yaml
recipe_dir: .

model:
  dir_or_tag: exp/${exp_tag}/model_pack
  trust_user_code: false
  call_args: {}

ui:
  app_script: src/app.py
  title: ${set_corpus_and_system:} demo
  description: README.md
  inputs:
    - key: speech
      type: audio
      label: "Input Audio"
  outputs:
    - key: hyp
      type: text
      label: "Transcription"

pack:
  out_dir: demo
  include: []
  exclude: []
  requirements:
    - git+https://github.com/espnet/espnet.git
  readme: ${config_path:../src/hf_demo_readme.md}
  readme_context:
    title: ${ui.title}
    sdk: gradio
    app_file: app.py

upload_demo:
  hf_repo: espnet/${set_corpus_and_system:}_${exp_tag}
  update: false
  delete_patterns:
    - "*"
```

## `model`

`model.dir_or_tag` selects the packed model the demo runs against:

- a local packed directory (relative to the demo dir), e.g. `../model_pack`
- a remote tag understood by `espnet_model_zoo`

`model.call_args` are call-time-only kwargs (`model(sample, **call_args)`).
Initialization-time overrides are not supported here; put those in the
packed `conf/inference.yaml` before running `pack_model`. See
[Publication configuration](./publish_config.html) for how a model gets
packed.

## `ui`

`ui.app_script` is copied into the packed demo as `app.py`. `ui.description`
can be a markdown/text file path (copied into the bundle) or inline text.

`ui.inputs`/`ui.outputs` entries have:

| Key | Description |
| --- | --- |
| `key` | model-facing field name (input key or output dict key) |
| `type` | a registered UI asset type — built-in: `audio`, `text` |
| `label` | UI label shown in Gradio |

The default `app.py` binds components positionally, in the order they appear
in `inputs`/`outputs`, so keep that order aligned with what the packed
inference callable expects and returns.

## `pack`

| Key | Description |
| --- | --- |
| `out_dir` | packed demo output directory (default `demo`) |
| `include` | extra files/dirs to copy into the bundle (globs supported) |
| `exclude` | glob patterns removed from copied `include` paths |
| `requirements` | pip specifiers or `git+https://...` entries written to `requirements.txt` |
| `readme` | Hugging Face Space `README.md` template path |
| `readme_context` | values overriding the template context inferred by `pack_demo` (e.g. `title`, `emoji`, `sdk`, `tags`) |

## `upload_demo`

Minimal example:

```yaml
upload_demo:
  hf_repo: yourname/your-demo
```

| Key | Description |
| --- | --- |
| `hf_repo` | required; full repo id, e.g. `yourname/your-demo` |
| `organization` | prefixed onto `hf_repo` when it has no `/` |
| `repo_type` | HF repo type (default `space`) |
| `update` | allow uploading over an existing Space (default `false`) |
| `delete_patterns` | glob patterns of existing Space files to delete first when `update: true` (default `["*"]`) |
| `create` | extra kwargs forwarded to `HfApi.create_repo` |

Current behavior: `upload_demo` uploads `pack.out_dir` (the packed demo
directory produced by `pack_demo`) to `hf_repo`, and raises if that directory
does not exist or if the Space already exists and `update` is not `true`.

## Related pages

- [Demo guide](../stages/demo.html)
- [Publication configuration](./publish_config.html)
- [Inference configuration](./infer_config.html)
