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

## 1. Run

`pack_model` needs `training_config` (to resolve `exp_tag`/`exp_dir`) and
`publication_config`. Running `infer` and `measure` first is recommended so
the bundle can include the inference results and `metrics.json` in the
README:

```bash
python run.py --stages pack_model \
  --training_config conf/tuning/training_e_branchformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

`run.py` propagates the training identity (`exp_tag`, `exp_dir`) into
`publication_config`, and propagates `inference_dir` from `inference_config`
into `publication_config` so `pack_model` can find `metrics.json` under it.
Add `upload_model` to `--stages` (with the same flags) to also push the
bundle to Hugging Face Hub:

```bash
python run.py --stages pack_model upload_model \
  --training_config conf/tuning/training_e_branchformer.yaml \
  --publication_config conf/publication.yaml
```

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
4. **Configs** — the training/inference/metrics/publication configs are
   written into `conf/` with paths rewritten to be bundle-relative.
5. **`meta.yaml`** is written at the bundle root, recording the copied
   `files`/`yaml_files` and other bundle metadata consumed by
   `InferenceModel.from_packed()`.

Add recipe-local Python code to `pack_model.include` when the packed
`conf/inference.yaml` refers to it.

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
│   ├── inference.yaml     # only if --inference_config was passed
│   ├── metrics.yaml       # only if --metrics_config was passed
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
model = InferenceModel.from_packed("exp/my_run/model_pack")

# From a model uploaded to Hugging Face Hub via upload_model:
model = InferenceModel.from_pretrained("espnet/my-model", trust_user_code=True)

result = model(audio_array)
batch_result = model.forward_batch([audio_a, audio_b])
```

`InferenceModel` loads the packed `conf/inference.yaml` (located through
`meta.yaml`), reconstructs the configured provider/runner backend, and calls
the same optional recipe `output_fn` used during inference. Set
`trust_user_code=True` only when the bundled recipe code (e.g. `src/`) is
intentionally trusted, since it is imported from the bundle at load time.

## 5. `upload_model`

`upload_model` uploads `pack_model.out_dir` to Hugging Face Hub.

Required field:

- `publication_config.upload_model.hf_repo`

The packed directory must already exist (run `pack_model` first). By
default `update: false`, so uploading over an existing repo raises an
error; set `upload_model.update: true` to upload over it.

## Related pages

**Stage API:** [`pack_model`](../../guide/espnet3/utils/pack_model.html),
[`upload_model`](../../guide/espnet3/utils/upload_model.html), and
[`InferenceModel`](../../guide/espnet3/publication/InferenceModel.html).
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
