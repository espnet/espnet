---
title: ESPnet3 Demo Configuration
---

# ESPnet3 Demo Configuration

`demo.yaml` describes a UI around a **packed model**. It does not instantiate
a training model or point directly at an unpacked `inference.yaml`.

```bash
python run.py --stages pack_demo \
  --training_config conf/training_asr_transformer.yaml \
  --demo_config conf/demo.yaml
```

The packed model must already exist. The training role supplies experiment
identity for defaults such as `exp/${exp_tag}/model_pack`. To run without
training YAML, supply `--exp_dir` and explicit model/UI/output paths that do
not depend on missing inherited keys.

## Required model source

Set `model.dir_or_tag` in demo YAML. Neither `pack_model.out_dir` nor
`upload_model.hf_repo` automatically populates it:

```yaml
model:
  dir_or_tag: exp/training_asr_transformer/model_pack
  device: auto
```

A local directory is resolved and linked into the demo bundle as
`model_pack`; the runtime configuration uses that local bundle reference.
A remote `owner/repository` is loaded by `InferenceModel.from_pretrained()`.
Keep an external local model directory available while using a symlinked
local demo. A Space upload must contain the referenced model files.

`model.trust_user_code` defaults to `false`. Set it to `true` when the model
bundle references recipe code or `contract.imports`. It is forwarded to
`InferenceModel`; it is not a top-level inference-model constructor flag in
`inference.yaml`.

```yaml
model:
  dir_or_tag: yourname/your-model
  trust_user_code: true
  call_args:
    temperature: 0.7
```

`model.call_args` are constant call-time kwargs. Backend construction options
belong in the packed inference config, not here. `DemoSession` loads the
configured inference runner through `InferenceModel`, so the same runner
formatting and contract validation apply.

## Contract-generated UI

By default, inputs and outputs come from the packed inference contract.
The ASR contract declares `speech` input and `text` output; the generated
widget names follow those fields. Explicit UI lists replace the respective
inferred lists:

```yaml
ui:
  app_script: ${config_path:../src/app.py}
  title: ASR demo
  description: README.md
  inputs:
    - key: speech
      type: audio
      label: Input Audio
  outputs:
    - key: text
      type: text
      label: Transcription
```

Each `key` is a model-facing input/output name. Components are passed to the
inference function in UI input order and outputs are selected by key.
Built-in UI assets are `audio` and `text`. The `number` and `segments`
contract kinds need a custom UI asset or an explicit compatible UI mapping.
Custom kinds need a matching asset registered on `session.registry` before
components are built. Include custom Python modules via `pack.include`, and
use absolute imports in the app.

`ui.description` can be inline text or a file. Prefer a recipe-relative
existing file such as `README.md`; keep the path and copied destination
consistent when customizing the packer.

## TEMPLATE values and packing

The ASR TEMPLATE is `egs3/TEMPLATE/esp2_asr/conf/demo.yaml`.

| Key | TEMPLATE or runtime default |
| --- | --- |
| `model.dir_or_tag` | `exp/${exp_tag}/model_pack` |
| `model.device` | `auto` |
| `model.trust_user_code` | `false` at runtime |
| `model.call_args` | empty mapping at runtime |
| `ui.app_script` | `${config_path:../src/app.py}` |
| `ui.title` | `${set_corpus_and_system:} demo` |
| `ui.description` | `README.md` |
| `pack.out_dir` | `demo` |
| `pack.include`, `pack.exclude` | empty lists |
| `pack.requirements` | `espnet[asr,demo] @ git+https://github.com/espnet/espnet.git` |
| `pack.readme` | `${config_path:../src/hf_demo_readme.md}` |

```yaml
pack:
  out_dir: exp/demo_asr
  include:
    - src/my_ui.py
    - assets
  exclude:
    - "**/*.log"
  requirements:
    - espnet[asr,demo] @ git+https://github.com/espnet/espnet.git
  readme: ${config_path:../src/hf_demo_readme.md}
  readme_context:
    title: ASR demo
    sdk: gradio
    app_file: app.py
```

The packer writes `demo.yaml`, `app.py`, requirements (when nonempty), README,
and included assets. The reviewed implementation reuses an existing output
directory without cleaning removed assets and cannot safely recreate an
existing `model_pack` symlink. Pack into a fresh directory when rebuilding.
Unlike model packing, there is no `pack.allow_overwrite` switch.

## Space upload

```yaml
upload_demo:
  hf_repo: yourname/your-demo
  private: false
  update: false
  delete_patterns:
    - "*"
```

```bash
python run.py --stages upload_demo \
  --training_config conf/training_asr_transformer.yaml \
  --demo_config conf/demo.yaml
```

The upload uses the configured `pack.out_dir` and the local Hugging Face login
(`hf auth login`). `hf_repo` is required; `organization` prefixes a repository
name with no slash. `repo_type` defaults to `space`; `create` forwards extra
repository-creation kwargs. The default `update: false` refuses an existing
Space. With `update: true`, `delete_patterns` defaults to `"*"` (full
replacement); an empty list makes the upload additive.

See [publication configuration](./publish_config.html) for model bundles,
and [contracts](./contract.html) for field-driven UI semantics.
