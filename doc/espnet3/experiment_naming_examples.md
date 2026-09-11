---
title: Getting Started with ESPnet3
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-04-24
---

# ESPnet3 Experiment Naming Examples

Below are some examples of how experiment names are determined based on the YAML configs.

::: note
`${self_name:}` is a resolver: at config-load time it is rewritten to the **stem of the
config file currently being loaded** (e.g. `training.yaml` → `training`,
`training_e_branchformer.yaml` → `training_e_branchformer`). It is not tied to `exp_tag` —
it is just the filename. Source:
[`espnet3/utils/config_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/config_utils.py).
:::

---

### Example 1: training-driven naming

The shipped `TEMPLATE/asr/conf/training.yaml` uses the resolver by default:

```yaml
exp_tag: ${self_name:}
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

So running `--training_config conf/training.yaml` resolves `exp_tag` to `training`
and writes outputs under:

```text
exp/training/
```

If the same `run.py` invocation also passes `--inference_config`,
`apply_training_experiment_context()` copies `exp_tag` and `exp_dir` from
`training_config` into `inference_config` (overwriting any conflicting value there, with a
warning). `TEMPLATE/asr/conf/inference.yaml` then resolves:

```yaml
inference_dir: ${exp_dir}/${self_name:}
```

Here `${self_name:}` is the stem of `inference.yaml` itself (`inference`), so the result is:

```text
exp/training/inference
```

---

### Example 2: standalone inference naming

If inference runs on its own (no `--training_config` in the same invocation),
`inference.yaml` must carry its own identity — its shipped default `exp_tag:` is blank:

```yaml
exp_tag: whisper_eval
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

That produces:

```text
exp/whisper_eval/inference
```

---

### Example 3: naming a tuning variant

A common convention is one training config per variant under `conf/tuning/`, each relying on
the `${self_name:}` default instead of hand-typing a tag:

```text
conf/
  tuning/
    training_e_branchformer.yaml   # exp_tag: ${self_name:}  (inherited from training.yaml)
  inference_beam5.yaml             # exp_tag: ${self_name:}
```

Because `${self_name:}` resolves independently for each file, running each config on its own
produces:

```text
exp/training_e_branchformer/
exp/inference_beam5/inference_beam5/
```

(`inference_beam5/inference_beam5` because `inference_dir: ${exp_dir}/${self_name:}` resolves
`self_name` a second time, against the `inference_beam5.yaml` filename.)

If `inference_beam5.yaml` is instead run together with `training_e_branchformer.yaml` via
`--training_config`/`--inference_config` in the same `run.py` call, the training config's
`exp_tag`/`exp_dir` win (see Example 1), so the decoding outputs move under the training
experiment directory: `exp/training_e_branchformer/inference_beam5/`.