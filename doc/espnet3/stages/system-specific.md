---
title: ESPnet3 System-specific Stages
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 System-specific Stages

This page is for people who develop ESPnet3 **System** classes (i.e., authoring
or extending `espnet3/systems/*`), not for day-to-day recipe users.

ESPnet3 stage execution is driven by a **System** class (typically a subclass of
`espnet3.systems.base.system.BaseSystem`). The System provides stage methods
(e.g., `train()`, `infer()`) and can define **system-specific stages** that are
not part of the generic pipeline.

## What is a "system-specific stage"?

A system-specific stage is any method on your System that you add to the stage
list in your recipe's `run.py`.

Examples:

- Tokenizer training (`train_tokenizer`)
- Task-specific preprocessing (`prepare_labels`, `dump_features`)
- Export (`export_onnx`, `export_torchscript`)
- Custom evaluation (`evaluate_custom`)

## How stages are discovered and executed

Recipe `run.py` defines the available stage names and passes them to the
argument parser via `--stages`. Internally, stage names are resolved and then
invoked by `espnet3.utils.stages.run_stages()` as **methods on the System
instance**.

In other words:

- If `--stages foo` is requested, the System must implement `def foo(self): ...`.
- Stage settings should live in YAML configs; stages are called without extra
  CLI arguments.

## Adding a new stage in your recipe

1. **Implement a method on your System**

   ```python
   class MySystem(BaseSystem):
       def train_tokenizer(self):
           ...
   ```

2. **Expose it in your recipe `run.py` stage list**

   Add your stage name to the `DEFAULT_STAGES` (or equivalent) list in your
   recipe `run.py` so it becomes selectable by `--stages`.

3. **Run it**

   ```bash
   python run.py --stages train_tokenizer --train_config conf/train.yaml
   ```

## Config requirements (and adding new configs)

Which stages require which config files is not "magically" inferred from the
System class. Instead, it is enforced by each recipe's `run.py`.

Typical pattern: start by reading the template runner at `egs3/TEMPLATE/asr/run.py`.

At a high level:

- CLI flags load optional configs (`--train_config`, `--infer_config`, ...).
- `run.py` defines a mapping from stage name → required config object.
- If a requested stage is missing its required config, `run.py` raises early.

### Checkpoints (with `run.py` snippets)

When a new system-specific stage needs its own config (instead of reusing
`train.yaml`/`infer.yaml`/etc.), follow this checklist.

1. **Add a new CLI flag and load the config**

   ```python
   # build_parser(...)
   parser.add_argument(
       "--export_config",
       default=None,
       type=Path,
       help="Hydra config for export-related stages.",
   )

   # main(...)
   export_config = (
       None
       if args.export_config is None
       else load_config_with_defaults(args.export_config)
   )
   ```

2. **Pass it into your System**

   ```python
   system = system_cls(
       train_config=train_config,
       infer_config=infer_config,
       measure_config=measure_config,
       publish_config=publish_config,
       demo_config=demo_config,
       export_config=export_config,  # new
   )
   ```

3. **Store it on the System**

   ```python
   class MySystem(BaseSystem):
       def __init__(self, *, export_config=None, **kwargs):
           super().__init__(**kwargs)
           self.export_config = export_config
   ```

4. **Enforce "stage requires config"**

   ```python
   required_configs = {
       "create_dataset": train_config,
       "collect_stats": train_config,
       "train": train_config,
       "infer": infer_config,
       "measure": measure_config,
       "pack_model": train_config,
       "upload_model": publish_config,
       "pack_demo": demo_config,
       "upload_demo": demo_config,
       "export": export_config,  # new requirement
   }

   missing = [
       s for s in stages_to_run if s in required_configs and required_configs[s] is None
   ]
   if missing:
       raise ValueError(
           f"Config not provided for stage(s): {', '.join(missing)}. "
           "Use the matching --*_config flag."
       )
   ```

5. **Log config metadata per stage**

   The template runner wires an `on_stage_start` hook into
   `espnet3.utils.stages.run_stages()`. Extend that hook to include your new
   config when relevant.

   ```python
   run_stages(
       system=system,
       stages_to_run=stages_to_run,
       on_stage_start=lambda stage, log: _log_stage_metadata(
           log,
           train_config=train_config,
           infer_config=infer_config,
           measure_config=measure_config,
           publish_config=publish_config,
           demo_config=demo_config,
           export_config=export_config,  # new
       ),
   )
   ```

   A common pattern inside `_log_stage_metadata()` is to log the resolved YAML
   for each config:

   ```python
   from omegaconf import OmegaConf

   if train_config is not None:
       logger.info(
           "Train config content:\n%s",
           OmegaConf.to_yaml(train_config, resolve=True),
       )
   ```

6. **Add docs + an example `conf/*.yaml`**

   Create a new config file under your recipe (`egs3/<recipe>/<task>/conf/`)
   and link it from the relevant stage doc under `doc/vuepress/src/espnet3/stages/`.

## Recommended conventions

- Use short, verb-style snake_case stage names (e.g., `train_tokenizer`,
  `export_onnx`).
- Keep stages idempotent when possible (safe to re-run).
- Reuse the standard stages (`create_dataset`, `collect_stats`, `train`, `infer`,
  `measure`) and add only what your task needs.

## Related docs

- [Systems overview](../core/systems.md)
- [Stage configs](../config/index.md)
- Stages: [train](./train.md), [infer](./inference.md), [measure](./measure.md)
