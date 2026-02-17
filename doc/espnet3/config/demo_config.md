---
title: ESPnet3 Demo Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Demo Configuration

This page explains the `demo.yaml` schema used by demo packaging and upload.
It ties a demo UI to an inference config and runtime settings.
For a deeper dive on demo behavior and customization, see
[Demo guide](../stages/demo.md).

## Minimum required keys (typical demo pack)

Required:

- `system`
- `infer_config`
- `output_keys` (required when `ui.outputs` are defined)

Common optional:

- `ui` (if you do not rely on system defaults)
- `inference` (custom provider/runner classes)
- `pack` (output folder and bundled files)
- `upload_demo` (only for uploads)

Minimal example:

```yaml
system: asr
infer_config: conf/infer.yaml

output_keys:
  text: hyp
```

## ✅ Config sections overview

| Section | Description |
| --- | --- |
| `system` | System name (e.g., asr) used for default provider/runner resolution. |
| `infer_config` | Path to the inference config used by the demo runtime. |
| `output_keys` | UI output mapping for inference results. |
| `ui` | Gradio UI config (optional). |
| `inference` | Provider/runner overrides (optional) via import path. |
| `extra_kwargs` | Constant runner kwargs passed to inference. |
| `pack` | Files to bundle in the demo package. |
| `upload_demo` | Upload destination for the demo (e.g., HF Space). |

## Core config layout

```yaml
system: asr
infer_config: conf/infer.yaml

output_keys:
  text: hyp

pack:
  out_dir: demo
  files:
    - exp/your_exp_dir/last.ckpt
    - exp/your_exp_dir/config.yaml

upload_demo:
  hf_username: your-hf-username
  repo_name: your-demo
  repo_type: space
  space_sdk: gradio
```

## Optional UI and inference overrides

If you want to override the default UI or inference classes, add:

```yaml
ui:
  title: "ESPnet3 Demo"
  inputs:
    - name: speech
      type: audio
  outputs:
    - name: text
      type: textbox

inference:
  provider_class: your.module.Provider
  runner_class: your.module.Runner

extra_kwargs:
  beam_size: 8
```
