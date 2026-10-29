---
title: ESPnet3 Publication Stages
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-26
---

# ESPnet3 Publication Stages

The publication stage packages a trained model for distribution, and
optionally uploads it to Hugging Face Hub.

This is a two-step process:

| Step           | Description                        | Implementation                              |
| -------------- | ----------------------------------- | -------------------------------------------- |
| `pack_model`   | Builds a self-contained bundle.     | `espnet3.utils.publication_utils.pack_model`  |
| `upload_model` | Uploads the bundle to Hugging Face. | `espnet3.utils.publication_utils.upload_model`|

## Pack without rerunning training

`BaseSystem.pack_model()` needs the resolved `train` entry, and uses `infer` and
`measure` entries when available. The current CLI only retains explicitly
requested entries. Therefore `--stages pack_model --training_config ...` alone
does not retain `train`, even though it loads the training role.

To package existing artifacts without executing training or inference again,
construct the system directly. From `egs3/mini_an4/esp2_asr`, after training,
inference, and measurement have succeeded:

```python
from pathlib import Path

from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.utils.config_utils import load_and_merge_config

paths = {
    "training": "conf/training_asr_transformer.yaml",
    "inference": "conf/inference.yaml",
    "metrics": "conf/metrics.yaml",
    "publication": "conf/publication.yaml",
}
configs = {
    role: load_and_merge_config(
        path,
        config_name=f"{role}.yaml",
        default_package="egs3.TEMPLATE.esp2_asr",
        resolve=False,
    )
    for role, path in paths.items()
}
system = ASRSystem(
    configs=configs,
    exp_dir=Path("exp/training_asr_transformer"),
    stages_to_run=("pack_model",),
)
system.pack_model()
```

The constructor builds independent configs for the prefix through `pack_model`;
calling this method executes only packing. It does not run the prefix stages or
write the launcher's stage logs/snapshots. The same pattern with
`stages_to_run=("measure",)` and `system.measure()` scores existing inference
outputs without invoking inference. A complete CLI pipeline may request all
predecessors plus `pack_model`, but doing so also executes those stages.

To upload an already packed bundle, explicitly choose the upload stage and its
role. The training config or saved experiment directory supplies identity:

```bash
python run.py --stages upload_model \
  --training_config conf/training_asr_transformer.yaml \
  --publication_config conf/publication.yaml
```

Uploading is a separate network write and uses the configured destination and
local Hugging Face authentication.

## 2. Configuration

The stage is configured in `conf/publication.yaml`, under the `pack_model`
and `upload_model` sections:

| Section                                  | Description                                                     |
| ----------------------------------------- | ---------------------------------------------------------------- |
| `pack_model.out_dir`                      | output bundle directory (default `${exp_dir}/model_pack`)         |
| `pack_model.allow_overwrite`               | overwrite an existing `out_dir` (default `false`)                 |
| `pack_model.include`, `exclude`            | extra paths to copy, and patterns skipped during the bulk copy   |
| `pack_model.files`, `yaml_files`           | named artifacts, always copied and registered in `meta.yaml`      |
| `pack_model.include_model_detail`          | include `repr(model)` in the README                               |
| `pack_model.readme`                       | README template path                                              |
| `upload_model.hf_repo`                     | target Hugging Face repo, e.g. `espnet/my-model`                  |
| `upload_model.update`                      | allow uploading over an existing repo (default `false`)          |

For a full list of options, see [Publication Configuration](../config/publish_config.html).

## 3. `pack_model`

`pack_model` builds the bundle in this order:

1. **Bulk copy** — copies `${exp_dir}` into the bundle, then each path in
   `include`. `exclude` patterns apply only to this step.
2. **Named artifacts (`files`)** — copies each entry individually and
   registers it in `meta.yaml`; `exclude` does not apply here.
3. **Named YAML artifacts (`yaml_files`)** — same as `files`, but paths
   inside the YAML are rewritten to bundle-relative form.
4. **Configs** — the available training/inference/metrics/publication configs are
   written into `conf/` with paths rewritten to be bundle-relative.
5. **`meta.yaml`** is written at the bundle root, recording the copied
   `files`/`yaml_files` and other bundle metadata consumed by
   `InferenceModel.from_packed()`.

Include recipe-local Python code and data metadata needed by the inference
provider via `pack_model.include` (`src/`, `dataset/`, or other actual dependencies).
Keep paths relative to the recipe working directory, or use explicit absolute
paths: non-glob include/named-artifact paths in this implementation can depend on
the process working directory. Verify loading from outside the recipe directory.

::: important
`pack_model.allow_overwrite: true` runs an unguarded `shutil.rmtree(out_dir)`
before repacking. Keep `out_dir` pointed at a dedicated subdirectory such as
the default `${exp_dir}/model_pack` — never at `exp_dir` or the recipe root
itself — since nothing currently checks that `out_dir` isn't an ancestor of
`exp_dir`
([`publication_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/publication_utils.py)'s
`pack_model`).
:::

A typical packed bundle looks like:

```text
model_pack/
├── conf/
│   ├── training.yaml
│   ├── inference.yaml     # when an infer stage config is available
│   ├── metrics.yaml       # when a measure stage config is available
│   └── publication.yaml
├── exp/            # copied `exp_dir` contents (checkpoints, logs, ...)
├── src/            # if included via `pack_model.include`
├── metrics.json    # copied from inference_dir, if `measure` already ran
├── meta.yaml
└── README.md
```

## 4. Packaged model inference

The external consumer of a packed bundle is
`espnet3.publication.InferenceModel`:

```python
from espnet3.publication import InferenceModel

# From a local directory produced by pack_model:
model = InferenceModel.from_packed("exp/my_run/model_pack", trust_user_code=True)

# From a model uploaded to Hugging Face Hub via upload_model:
model = InferenceModel.from_pretrained("espnet/my-model", trust_user_code=True)

import numpy as np

speech = (16000, np.zeros(16000, dtype=np.float32))
result = model(speech)
batch_result = model.forward_batch([speech, speech])
```

`InferenceModel` loads the packed `conf/inference.yaml` (located through
`meta.yaml`), reconstructs the configured provider/runner backend, and applies
the inference contract for validation and processing (resample, channel check,
output validation). It then calls the same optional recipe `output_fn` used during inference for legacy recipes. Set
`trust_user_code=True` when the packed `inference.yaml` references bundled
`src.*` modules (e.g., custom kinds/normalizers via `contract.imports`); the
bundle root is then added to `sys.path`.

## 5. `upload_model`

`upload_model` uploads `pack_model.out_dir` to Hugging Face Hub.

Required field:

- `publication_config.upload_model.hf_repo`

The packed directory must already exist (run `pack_model` first). By
default `update: false`, so uploading over an existing repo raises an
error; set `upload_model.update: true` to upload over it.

## Related pages

**Stage API:** [`pack_model`](https://github.com/espnet/espnet/blob/master/espnet3/utils/publication_utils.py),
[`upload_model`](https://github.com/espnet/espnet/blob/master/espnet3/utils/publication_utils.py), and
[`InferenceModel`](https://github.com/espnet/espnet/blob/master/espnet3/publication/inference_model.py).
Run these after [training](./train.html), normally after [measurement](./metrics.html);
then use [demo packaging](./demo.html) when publishing an interactive UI.

<DocCards :cols="3">
  <DocCard
    title="Publication configuration"
    desc="All options for configuring the publication stage"
    icon="tabler:file-code"
    href="../config/publish_config.html"
  />
</DocCards>
