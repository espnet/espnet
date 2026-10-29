---
title: ESPnet3 Experiment Naming Examples
---

# Experiment naming and stage configuration

The shared launcher loads a selected role YAML over the matching
`egs3.TEMPLATE.esp2_asr/conf/<role>.yaml` defaults with resolution deferred.
`${self_name:}` is substituted while loading each file, using that file's stem.
This is not a mutable global resolver that always follows the training name.

## Training names

The TEMPLATE training defaults include:

```yaml
exp_tag: ${self_name:}
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
```

Loading `conf/training_small.yaml` therefore gives `exp_tag: training_small`
and `exp/training_small`, unless that file explicitly overrides the naming.
The default statistics directory is shared across variants. Set
`stats_dir: ${exp_dir}/stats` when different data/frontends need separate stats.

## Inference inherits earlier stage configuration

The inference defaults contain:

```yaml
inference_dir: ${exp_dir}/${self_name:}
```

`exp_dir` comes from the merged earlier-stage configuration or saved snapshots,
not from `apply_training_experiment_context()` (which is not used by the shared
launcher). Role configs are merged at the top level in declared stage order;
an explicitly supplied role overrides the corresponding earlier values.
Conflicting saved/explicit experiment identities can raise `ConfigError` rather
than being silently overwritten with a warning.

```bash
python run.py --stages infer \
  --exp_dir exp/training_small \
  --inference_config conf/inference_beam5.yaml
```

This selects the earlier experiment and writes `exp/training_small/inference_beam5`
when `inference_dir` is inherited from the inference defaults. If the inference
YAML itself sets `inference_dir`, that explicit value wins.

## Explicit standalone identity

An evaluation-only config may provide its own complete identity instead:

```yaml
recipe_dir: .
exp_tag: whisper_eval
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}
```

It must also supply a compatible model/backend and dataset. A blank `exp_tag`
is not a usable experiment name. Do not assume supplying a later role config
creates or loads a trained model automatically.

## Supported CLI overrides

`--exp_dir` and stage-directory flags derived from `StageSpec.log_dir` are
supported. Each role flag takes one path. Arbitrary Hydra `key=value` CLI
overrides and the removed `--config`/numbered-stage interface are not supported.

The baseline launcher keeps only requested stage config entries. To run ASR
training through the CLI, request `train_tokenizer` alongside `train`; to score
dataset-backed references, request `infer` alongside `measure`. Direct system
construction can build predecessor configs without executing those methods.
