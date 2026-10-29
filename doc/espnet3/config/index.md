# Config files

ESPnet3 recipes use YAML files under `conf/`. Config **roles** are declared by
`StageSpec.config` on the system, and `launch()` generates one
`--<role>_config` option for each role. The ASR roles are `training`,
`inference`, `metrics`, `publication`, and `demo`.

| Role | File | Stages that declare it |
| --- | --- | --- |
| `training` | `training.yaml` | `create_dataset`, `train_tokenizer`, `collect_stats`, `train`; also `pack_model` |
| `inference` | `inference.yaml` | `infer` |
| `metrics` | `metrics.yaml` | `measure` |
| `publication` | `publication.yaml` | `pack_model`, `upload_model` |
| `demo` | `demo.yaml` | `pack_demo`, `upload_demo` |

`pack_model` declares both `training` and `publication`, so a stage does not
necessarily read exactly one role. Custom systems may declare other roles.

## Recipe entry point

```python
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch


if __name__ == "__main__":
    launch(ASRSystem)
```

Keep the `__main__` guard: worker processes can re-import the entry point.
The default configuration package is inferred from the system class's module:
`ASRSystem` uses `egs3.TEMPLATE.esp2_asr`. A recipe may pass
`conf_package="my_project.recipe_defaults"` to `launch()` explicitly.

## Loading and inheritance

When a role flag is supplied, its user file is deep-merged over that role's
TEMPLATE defaults with OmegaConf. A role flag that is omitted supplies no
explicit role config; it does not automatically load a recipe's `conf/` file.

Stage inheritance is different from the default/user-file merge: stages are
considered in declared order, and later stages replace **whole top-level
keys**. A stage's own keys replace inherited keys. Thus an inference `model:`
block replaces the training `model:` block; it does not extend that model.
Explicit inference `exp_tag` or `exp_dir` values also replace inherited ones.
There is no unconditional copying of training identity over explicit overrides.
Each resolved stage has its own `DictConfig` object.

```yaml
# conf/inference_beam5.yaml: a user override of TEMPLATE inference defaults.
# Omit experiment identity to inherit it from the training role.
model:
  asr_train_config: ${exp_dir}/config.yaml
  asr_model_file: ${exp_dir}/epoch0_step1_valid.loss.ckpt
  beam_size: 5
```

Use an actual checkpoint produced by your experiment; the filename above is
the tiny Mini-AN4 integration configuration's checkpoint.

## Launching stages

From `egs3/mini_an4/esp2_asr`, a complete local pipeline is:

```bash
python run.py \
  --stages create_dataset train_tokenizer collect_stats train infer measure pack_model \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

`--stages` selects stage bodies; it does not automatically execute their
prerequisites. `all` selects every declared stage, including upload stages,
so specify a stage list for a local run.

The reviewed launcher retains only explicitly requested entries in
`system.stage_configs`. Some stage methods access another stage's entry:
ASR `train` reads `train_tokenizer`, `measure` reads `infer`, and `pack_model`
looks up `train`, `infer`, and `measure`. Passing a role flag alone does not
retain that stage entry. The complete example above supplies these entries;
`--stages train` or `--stages measure` alone is not a supported shortcut on
this implementation. To score existing outputs, request `infer measure`;
the inference runner can reuse completed shards under its resume rules.

## Saved configurations and directory options

Before a stage body runs, `run_stages()` writes its own top-level keys to
`<exp_dir>/config/<stage>.yaml`. For a stage with a resolved log directory,
its full resolved snapshot is also written to
`<stage_log_dir>/<stage>.yaml`; logs and optional requirements live beside it.
Saved-config recovery searches the stage's log directory, the experiment
config directory, and the legacy layout. It does not rebuild missing datasets,
models, or prerequisites.

`--exp_dir` supplies an experiment directory for recovery. Generated
stage-directory flags use **hyphens**, such as `--inference-dir` and
`--stats-dir`; inspect `python run.py --help` for the actual system's options.
There are no arbitrary Hydra `key=value` command-line overrides. Edit a YAML
file instead. `--dry_run` skips stage bodies and bake/snapshot writes, but
system construction can still create `exp_dir`.

Dotted log-directory keys become safe argparse destinations. On this reviewed
launcher, forwarding those dotted destinations (for example
`tokenizer.save_path` and `pack.out_dir`) is incomplete; configure those paths
in YAML rather than relying on their generated directory flags. Directory
CLI overrides are also forwarded as `Path` objects on this baseline and can
fail YAML serialization later; YAML string paths are the reliable choice.

## Configuration reference

- [Training](./train_config.html): datasets, tokenizer, model, optimization.
- [Inference](./infer_config.html): backend, runner, outputs and batching.
- [Contract](./contract.html): inference, dataset and metric declarations.
- [Publication](./publish_config.html): packed models and upload settings.
- [Demo](./demo_config.html): packed-model UI and Space settings.
- [Parallel execution](../parallel.html): Dask runners and Lightning training.

The implementation references are `espnet3/systems/launch.py`,
`espnet3/utils/config_utils.py`, `espnet3/utils/stage_configs.py`, and
`espnet3/contract/stages.py`.
