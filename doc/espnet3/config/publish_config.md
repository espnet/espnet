---
title: ESPnet3 Publication Configuration
---

# ESPnet3 Publication Configuration

`publication.yaml` controls `pack_model` and `upload_model`. A model pack is a
recipe archive plus model artifacts, not just a checkpoint. It is consumed
by `espnet3.publication.InferenceModel` and by the demo.

## Stage/config requirements

`pack_model` declares config roles `training` and `publication`. Its method
also reads `system.stage_configs["train"]` and optionally `infer`/`measure`.
The reviewed `launch()` retains only explicitly requested stage entries;
passing `--training_config` with `--stages pack_model` alone does not retain
`train`. Use packing in a pipeline that includes those entries:

```bash
# From egs3/mini_an4/esp2_asr; also runs training and evaluation.
python run.py \
  --stages create_dataset train_tokenizer collect_stats train infer measure pack_model \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

This command is not an upload command. To pack an already trained experiment
without executing predecessor bodies, a Python caller can construct a system
with all required role configs, keep its predecessor stage-config entries,
and call `pack_model()` directly. Do not assume the standalone CLI reconstructs
those entries automatically on this baseline.

The training config supplies `recipe_dir`, `exp_tag`, `exp_dir`, and
`data_dir`. `inference_dir` comes from inference configuration/inheritance;
publication YAML has no independent default experiment identity. A publication
role may override inherited top-level keys under the normal
[stage inheritance rules](./index.html#loading-and-inheritance).

## Minimal user override

The default pack settings come from
`egs3/TEMPLATE/esp2_asr/conf/publication.yaml`:

```yaml
upload_model:
  hf_repo: yourname/your-model
```

The upload target is separate from `demo.yaml`'s `model.dir_or_tag`; setting
one does not fill in the other.

| Key | TEMPLATE default |
| --- | --- |
| `pack_model.out_dir` | `${exp_dir}/model_pack` |
| `pack_model.allow_overwrite` | `false` |
| `pack_model.inference_dir` | `${inference_dir}` |
| `pack_model.include_model_detail` | `false` |
| `pack_model.readme` | `${config_path:../src/hf_model_readme.md}` |
| `pack_model.include` | `src`, `dataset` |
| `upload_model.private` | `false` |
| `upload_model.update` | `false` |
| `upload_model.delete_patterns` | `"*"`, used when updating |

## Copying and config serialization

The packer performs these steps:

1. Copy the experiment tree and extra `include` paths into the bundle,
   applying `exclude` only during these bulk copies.
2. Copy named `files` artifacts and record their relative paths in `meta.yaml`.
   These explicit entries are not filtered by `exclude`.
3. Copy `yaml_files` artifacts and rewrite their recipe paths.
4. Serialize the available training/inference/metrics/publication configs
   under `conf/`, rewriting recipe paths to bundle-relative form.
5. Write `meta.yaml`, the rendered README, and `metrics.json` when found.

```yaml
pack_model:
  out_dir: ${exp_dir}/model_pack
  allow_overwrite: false
  include:
    - ${recipe_dir}/src
    - ${recipe_dir}/dataset
    - ${data_dir}/**/bpe.model
    - ${data_dir}/**/bpe.vocab
    - ${data_dir}/**/tokens.txt
  exclude:
    - inference
    - inference_*
    - last.ckpt
    - "**/step*.ckpt"
    - "**/*.log"
    - "**/tensorboard/**"
    - "**/wandb/**"
```

This matches the include pattern used by
`egs3/mini_an4/esp2_asr/conf/publication.yaml`. Include recipe-local modules
needed to construct the published model, plus any tokenizer assets stored
outside the experiment tree. Prefer explicit `${recipe_dir}` paths: literal
relative include/file paths in the reviewed packer are resolved from the
process working directory, while glob handling uses the recipe root.

Named-artifact example:

```yaml
pack_model:
  files:
    bpemodel: ${data_dir}/bpe_5000/bpe.model
  yaml_files:
    tokenizer_config: ${data_dir}/bpe_5000/config.yaml
```

Do not overwrite a source directory with the bundle output. The reviewed
implementation removes an existing `out_dir` when `allow_overwrite: true`;
choose a dedicated output directory separate from source data/code.

Typical layout:

```text
model_pack/
  conf/
    training.yaml
    inference.yaml
    metrics.yaml
    publication.yaml
  src/
  dataset/
  exp/
  data/
  files/
  yaml_files/
  meta.yaml
  README.md
  metrics.json
```

For a direct caller that needs no stage bodies, the baseline constructor can
retain predecessor entries:

```python
from pathlib import Path

from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.utils.config_utils import load_and_merge_config


package = "egs3.TEMPLATE.esp2_asr"
paths = {
    "training": "conf/training_asr_transformer.yaml",
    "inference": "conf/inference.yaml",
    "metrics": "conf/metrics.yaml",
    "publication": "conf/publication.yaml",
}
configs = {
    role: load_and_merge_config(
        Path(path), config_name=f"{role}.yaml",
        default_package=package, resolve=False,
    )
    for role, path in paths.items()
}
system = ASRSystem(
    configs=configs, exp_dir=Path("exp/training_asr_transformer"),
    stages_to_run=("pack_model",),
)
system.pack_model()
```

Run that Python code from the recipe root after training/evaluation. It uses
the same defaults and unresolved role configs as the launcher, but it does
not replace `system.stage_configs` with a requested-stage-only mapping.
Only the final `pack_model()` method executes.

Only configs supplied to the packing function are serialized. In the system
wrapper they come from the retained `train`, `infer`, `measure`, and
`pack_model` stage entries, rather than directly from all role flags.
`InferenceModel` needs the packed inference config, so retain `infer` when
building a publicly usable inference bundle.

`metrics.json` is optional. The packer looks at top-level publication
`inference_dir`, then metrics and inference config locations,
and embeds a results table when available. `pack_model.readme_context`
overrides inferred README values; `include_model_detail: true` additionally
embeds the model's representation.

## Loading a bundle

```python
import soundfile as sf

from espnet3.publication import InferenceModel


model = InferenceModel.from_packed("exp/my_run/model_pack")
waveform, sample_rate = sf.read("sample.wav", dtype="float32")
result = model({"speech": (sample_rate, waveform)})
print(result["text"])
```

Use `from_pretrained("owner/repository")` for a Hugging Face bundle.
`from_packed()` validates `meta.yaml`, reads the inference config referenced
by `meta.yaml.yaml_files.inference_config`, and constructs its provider and
runner. The contract travels inside the packed inference YAML and applies
to both single and batch calls.

Passing `(sample_rate, waveform)` preserves the rate, unlike a bare array.
Soundfile stereo arrays are samples-first and the audio kind converts them
to channels-first. The backend or runner must handle the checked `Audio`
representation appropriately.

Set `trust_user_code=True` when the bundle contains needed recipe modules,
custom targets, or `contract.imports`; use the default `False` for bundles
that use only allowed built-in components. Conflicting imported recipe module
names are rejected rather than silently shadowing already loaded code.

```python
from espnet3.publication import InferenceModel


model = InferenceModel.from_pretrained(
    "yourname/custom-model", trust_user_code=True
)
```

## Uploading

```yaml
upload_model:
  hf_repo: yourname/your-model
  private: false
  update: false
  delete_patterns:
    - "*"
```

Once the model is packed, `upload_model` can run as a separate CLI stage:

```bash
python run.py --stages upload_model \
  --training_config conf/training_asr_transformer.yaml \
  --publication_config conf/publication.yaml
```

It uploads `pack_model.out_dir` using `huggingface_hub` and your local
`hf auth login`. The target repository is created as needed. Existing
repositories are refused unless `update: true`; updating deletes matches of
`delete_patterns` (default `"*"`) before upload. An empty list is additive.

Use `publication.yaml` / `--publication_config`, not `publish.yaml` /
`--publish_config`. See [demo configuration](./demo_config.html) for Spaces.
