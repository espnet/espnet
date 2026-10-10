---
title: ESPnet3 Demo Guide
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-28
---

# ESPnet3 Demo Guide

This page explains how to package and run an interactive Gradio demo from a
published ESPnet3 model.

A key advantage is that demos **reuse your packed model** through
`InferenceModel`, so the demo does not re-implement inference; it only wires
UI values to the same inference call used by `pack_model`.

## 1. Run

`pack_demo`'s default `model.dir_or_tag: exp/${exp_tag}/model_pack` needs
`exp_tag`, which is only resolved when `--training_config` is passed
alongside `--demo_config`:

```bash
python run.py --stages pack_demo \
  --training_config conf/tuning/training_e_branchformer.yaml \
  --demo_config conf/demo.yaml
```

::: important
Running `pack_demo`/`upload_demo` with `--demo_config` alone crashes with
`omegaconf.errors.InterpolationKeyError: Interpolation key 'exp_tag' not
found`, because `demo.yaml`'s default `model.dir_or_tag` references
`${exp_tag}` and nothing else supplies it. Always pass `--training_config`
(or set `model.dir_or_tag` to an explicit path/tag in `demo.yaml`).
:::

Add `upload_demo` to `--stages` (with the same flags) to push the packed
demo to a Hugging Face Space:

```bash
python run.py --stages pack_demo upload_demo \
  --training_config conf/tuning/training_e_branchformer.yaml \
  --demo_config conf/demo.yaml
```

::: important
Re-running `pack_demo` against the same `pack.out_dir` can fail with
`FileExistsError` if the packed model directory lives outside the demo
directory (the default TEMPLATE layout, where `model_pack` sits under
`exp/${exp_tag}/` and `demo` sits at the recipe root): `pack_demo`
creates `demo_dir/model_pack` as a symlink to the external model directory,
but does not remove a stale symlink left over from a previous run
([`packing.py`](https://github.com/espnet/espnet/blob/master/espnet3/publication/demo/packing.py)'s
`_link_local_model_into_bundle`). If a re-run fails this way, delete
`pack.out_dir` (or just `demo_dir/model_pack`) before running `pack_demo`
again.
:::

::: important
Stage logs for `pack_demo`/`upload_demo` are written directly into
`demo_config.pack.out_dir` — the same directory `upload_demo` uploads to the
Hugging Face Space. Passing `--write_requirements` writes a full `pip
freeze` snapshot into that same directory as `requirements.txt`, overwriting
the curated `requirements.txt` that `pack_demo` already wrote from
`pack.requirements`
([`system.py`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/system.py)'s
`stage_log_mapping`, and
[`logging_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/logging_utils.py)'s
`_write_requirements_snapshot`). Avoid `--write_requirements` with
`pack_demo`/`upload_demo`, and check `requirements.txt` in the packed
directory before uploading.
:::

## 2. Outputs

After `pack_demo`, ESPnet3 writes a runnable Gradio app into the output
directory (`demo.yaml`'s `pack.out_dir`; the TEMPLATE default is the
recipe-relative `demo` directory):

```bash
cd demo
python app.py
```

This starts a local Gradio server. Open the printed URL in your browser.

> [!IMPORTANT]
> `gradio` is required for local demo execution. Install it with `pip install gradio`.

A typical packed demo directory looks like:

```text
demo/
├── app.py            # copied from ui.app_script
├── demo.yaml         # resolved config, paths rewritten to be relative
├── model_pack -> ../model_pack/   # symlink, when the model lives outside demo_dir
├── README.md
└── requirements.txt  # written from pack.requirements
```

**After changing `app.py` or any UI asset, run `pack_demo` again** — the
packed directory is a snapshot copied at pack time, so edits to the source
`app.py`/assets are not picked up until the next `pack_demo`.

## 3. The three-layer demo design

The demo has three layers with separate responsibilities:

1. **`demo.yaml`** declares the model reference (`model.dir_or_tag`), model
   call-time kwargs (`model.call_args`), the UI input/output specs
   (`ui.inputs`/`ui.outputs`), and the app script to copy (`ui.app_script`).
2. **`load_demo_session()`** loads that config and builds a `DemoSession`
   (`espnet3.publication.demo.session`). The session resolves the packed
   model through `InferenceModel`, resolves UI assets from the registry, and
   builds the inference callable via `DemoSession.create_inference_fn()`.
3. **The packed `app.py`** owns the Gradio layout. The default app builds
   components from `session.input_specs`/`session.output_specs`, passes
   their values positionally to `session.create_inference_fn()`, and
   displays the returned values.

The runtime inference path is:

```text
Gradio values -> input spec keys -> InferenceModel(model_input, **call_args)
              -> result keys -> output spec values -> Gradio components
```

ESPnet3 owns model loading, packed-config resolution, input-key mapping,
call-time arguments, and the generic session callable. The recipe owns the
UI layout and any presentation-specific logic.

## 4. Configuring `demo.yaml`

For the full list of options, see [Demo configuration](../config/demo_config.html).

| Section              | Description                                                      |
| --------------------- | ----------------------------------------------------------------- |
| `model.dir_or_tag`     | packed model directory (relative to the demo dir) or HF Hub tag  |
| `model.call_args`      | kwargs passed to the model at call time, e.g. `beam_size`        |
| `ui.app_script`        | Gradio launcher script copied into the pack as `app.py`          |
| `ui.title`             | app title                                                        |
| `ui.description`       | markdown/text file path, or inline text                          |
| `ui.inputs`/`ui.outputs` | list of `{key, type, label}` UI component specs                |
| `pack.out_dir`         | packed demo output directory                                     |
| `pack.include`, `exclude` | extra files/dirs to copy, and exclusion patterns              |
| `pack.requirements`   | pip specifiers written to the packed `requirements.txt`          |
| `upload_demo.hf_repo` | target Hugging Face Space                                        |

For ordinary UI changes, edit `ui.inputs`/`ui.outputs` — each entry's `key`
must match the packed model's input/output contract — and reuse the
built-in `audio`/`text` asset types. Example:

```yaml
ui:
  app_script: src/app.py
  title: ASR demo
  inputs:
    - key: speech
      type: audio
      label: "Input Audio"
  outputs:
    - key: hyp
      type: text
      label: "Transcription"
```

For a new reusable component type, subclass `UIAsset`
(`espnet3.publication.demo.assets`) and register it in `DEFAULT_UI_ASSETS`.
For a one-off layout, edit or replace `ui.app_script` and call
`load_demo_session()`/`DemoSession` directly, keeping the positional order
of Gradio inputs/outputs aligned with `session.input_specs`/`output_specs`.

## Related pages

**Stage API:** [`pack_demo`](../../guide/espnet3/publication/pack_demo.html),
[`upload_demo`](../../guide/espnet3/publication/upload_demo.html),
[`DemoSession`](../../guide/espnet3/publication/DemoSession.html), and
[`UIAsset`](../../guide/espnet3/publication/UIAsset.html). Build a model bundle
with [publication stages](./publish.html) before packing a demo.

<DocCards :cols="3">
  <DocCard
    title="Demo configuration"
    desc="All options for configuring the demo stage"
    icon="tabler:file-code"
    href="../config/demo_config.html"
  />
  <DocCard
    title="Publication stages"
    desc="Pack and upload a model before packaging a demo around it."
    icon="tabler:package"
    href="publish.html"
  />
</DocCards>
