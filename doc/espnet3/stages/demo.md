---
title: ESPnet3 Demo Guide
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Demo Guide

This page explains how demo configs map to runtime behavior and how to
customize demos. A key advantage is that demos **reuse your existing inference
code** (providers/runners/models), so you do **not** need to write extra demo-
specific Python.

## Quick usage

### Run

```bash
python run.py --stages pack_demo upload_demo --demo_config conf/demo.yaml
```

### Run locally after packing

After `pack_demo`, ESPnet3 writes a runnable Gradio app into the output
directory (default is `demo/` if you set `pack.out_dir: demo`).

```bash
cd demo
python app.py
```

This starts a local Gradio server. Open the printed URL in your browser.

### Notes

- `gradio` is required for local demo execution: `pip install gradio`.

### Outputs

After packing, the output directory contains the runnable app, configs, and
links to your recipe assets. Example:

```text
demo/
├── app.py
├── config
│   └── infer.yaml
├── data -> ../data/
├── demo.yaml
├── exp -> ../exp/
├── README.md
└── requirements.txt
```

### Configure (in `demo.yaml`)

Keep the core settings in `demo.yaml`. For the full list, see
[Demo configuration](../config/demo_config.md).

| Config section | Description |
| -------------- | ----------- |
| `infer_config` | Path to the inference config used by the demo runtime. |
| `ui` | UI layout and component definitions. |
| `inputs` | Input field definitions and preprocessing mappings. |
| `outputs` | Output field definitions and postprocessing mappings. |

### UI configuration (Gradio)

UI is configured under `ui` in `demo.yaml`. The demo app is generated from this
config and wires inputs/outputs directly to your inference runner.

#### Resolution order

1. If `demo.yaml` includes `ui`, it is merged with system defaults from
   `build_ui_default()` (when available).
2. If `ui` is missing, ESPnet3 calls `build_ui(demo_cfg)` from
   `espnet3.systems.<system>.demo`.
3. If neither is available, `ui` is required in `demo.yaml`.

#### UI fields

| Field | Description |
| --- | --- |
| `title` | App title shown at the top of the demo page. |
| `description` | Markdown text shown under the title. |
| `article` | Optional Markdown section shown at the bottom. |
| `article_path` | Path to a Markdown file to load as `article`. If set, it overrides `article`. |
| `button.label` | Label for the Run button. |
| `inputs` | List of input component configs (`name`, `type`, and type-specific fields). |
| `outputs` | List of output component configs (`name`, `type`, and type-specific fields). |

### Notes

After `pack_demo`, the packed demo directory contains a `README.md` and the demo
UI renders it as the page article by default. To change what is shown in the
demo screen, edit `demo/README.md` in the packed directory (or set
`ui.article_path` to a different Markdown file).

Sample UI config:

```yaml
ui:
  title: "ESPnet3 Demo"
  description: "Run inference with your existing provider/runner."
  button:
    label: "Run"
  inputs:
    - name: speech
      type: audio
      sources: [mic, upload]
    - name: lang
      type: dropdown
      choices: [en, ja]
  outputs:
    - name: text
      type: textbox
      lines: 2
```

Topics to cover:

- How `demo.yaml` fields are used by the runtime and pack flow
- Overriding UI, provider, and runner classes
- Advanced customization patterns

## Developer Notes

### Supported component types

Each entry in `inputs`/`outputs` requires `name` and `type`. Supported `type`
values and key fields:

- `audio`: `sources` (mic/upload), `audio_type` (`numpy` by default)
- `textbox`: `lines`, `placeholder`
- `dropdown`: `choices`, `value`
- `number`: `value`
- `slider`: `min`, `max`, `step`, `value`
- `checkbox`: `value`
- `image`
- `file`

The component `name` becomes the key used by the demo runtime, so it must match
the expected inference input/output mapping.

For `audio` inputs, Gradio returns `(sample_rate, np.ndarray)`. The demo runtime
normalizes this to a `float32` waveform array before passing it to the runner.

UI input values are placed into a single-item dataset. The runner receives that
dataset and should read the inputs from `dataset[0]`. Only `extra_kwargs` from
`demo.yaml` are passed as `kwargs` to the runner.

Under the hood, the demo runtime packages those UI inputs into a **single-item
dataset**. This keeps the call pattern consistent with standard inference
runners (`forward(idx, dataset=..., model=...)`) while still letting you pass
simple UI fields.

Minimal demo dataset (conceptual behavior):

```python
class SingleItemDataset:
    def __init__(self, item):
        self._item = item

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        # Returns UI-defined fields (e.g., speech, lang) as a dict.
        return self._item
```

With this dataset, a runner can still use the familiar signature and pull
values from `dataset[0]`. Example runner implementation that matches the UI
sample above:

```python
from espnet3.parallel.base_runner import BaseRunner


class DemoRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **_):
        item = dataset[idx]
        hyp = model(item["speech"], lang=item.get("lang"))
        return {"hyp": hyp}
```

### Output mapping (`output_keys`)

When outputs are defined, `output_keys` maps UI output names to keys returned by
your runner/model result. This lets you return a structured dict (e.g.,
`{"hyp": ...}`) and map it to UI outputs. Example:

```yaml
output_keys:
  text: hyp
```

If `output_keys` is missing but outputs are defined, demo runtime raises an
error.

### System defaults (ASR example)

For ASR, defaults live in `espnet3/systems/asr/demo.py`:

- `build_ui_default()` defines the default input/output components.
- `build_ui(demo_cfg)` optionally modifies defaults using demo config.
- `build_inference_default()` defines default `output_keys` and `extra_kwargs`.

Minimal ASR UI defaults:

```python
def build_ui_default():
    return {
        "title": "ASR Demo",
        "inputs": [{"name": "speech", "type": "audio", "sources": ["mic", "upload"]}],
        "outputs": [{"name": "text", "type": "textbox"}],
    }
```
