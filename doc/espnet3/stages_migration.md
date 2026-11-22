## Migrating a system/recipe to the Stages Contract

This note covers migrating an existing ESPnet3 system or recipe `run.py` to
the stage contract introduced by `espnet3.contract.stages` and
`espnet3.systems.launch`.

### 1. Declare stages with `StageSpec`

A system declares its stages once, as a class attribute:

```python
from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    stages = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        # A stage needing more than one config role takes a tuple, not a
        # single string - both configs must then be given to run it.
        StageSpec(name="pack_model", config=("training", "publication")),
    )
```

Each `StageSpec.name` must match a callable method defined on the class.
`StageSpec.config` is the config role (or tuple of roles) the stage needs;
each role `"<role>"` names a `<role>_config` constructor argument. Roles
are not a fixed list - any non-empty identifier is valid, and a system's
full set of roles is derived from its own `stages` (`espnet3.contract.
stages.roles`).

Every public method the class defines (not starting with `_`) must be one
of the declared stage names - a system has no other public surface. This
is checked once, when the class is defined, by `check_stage_contract`
(`BaseSystem.__init_subclass__` calls it automatically).

### 2. Use `launch()` in the recipe `run.py`

```python
# egs3/mini_an4/esp2_asr/run.py
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ASRSystem)
```

`launch()` builds the CLI (`--stages`, one `--<role>_config` per declared
role, `--dry_run`, `--write_requirements`), loads each role's config,
constructs the system with every loaded role forwarded as `<role>_config`,
and runs the requested stages in declared order.

### 3. The `__main__` guard is required

`collect_stats`/`train` spawn Dask worker processes (and, on macOS,
`DataLoader` workers use `spawn` by default); each worker process
re-imports `run.py`. Without the guard, every worker re-runs the whole
pipeline from scratch, racing the parent process and failing with errors
such as `Shard is already locked by another runner`. Always wrap the
`launch(...)` call in `if __name__ == "__main__":`, exactly as shown
above - a bare module-level call is a bug, not a style choice.

### 4. Removed interfaces

The following are gone, with no compatibility shim:

- `BaseSystem(stage_log_mapping=...)` and `system.stage_log_dirs` - a
  stage's log directory is now resolved from its own `StageSpec.log_dir`
  (`espnet3.contract.stages.stage_log_dir`), not a mapping passed in.
- `espnet3.utils.stages_utils.parse_cli_and_stage_args` - replaced by
  `espnet3.systems.launch.build_parser` / `launch`.
- `build_parser` / `main` / `DEFAULT_STAGES` previously in
  `egs3/TEMPLATE/esp2_asr/run.py` - that file is gone; recipes call
  `launch()` directly as shown above.

If your recipe or an open PR still imports any of these, update it to the
`StageSpec`/`launch()` pattern above.

### 5. Module path changes under `src/`

System-internal modules that are not part of the system's own public
contract (`system.py`, `inference.py`, `task.py`, `metrics/`) moved under
a `src/` subpackage, with no re-export shim:

- `espnet3.systems.esp2_asr.tokenizers.*` -> `espnet3.systems.esp2_asr.src.tokenizers.*`
- `espnet3.systems.esp2_asr.transducer_task` -> `espnet3.systems.esp2_asr.src.transducer_task`
- `espnet3.systems.f5tts.{cfm,dit,f5tts,modules,pinyin,preprocessor,remove_long_short,rotary,solvers,utils,vocoder_mel,create_token_list}` -> the same names under `espnet3.systems.f5tts.src.*`

Update any `task:` or Hydra `_target_:` dotted string in a user config
that points at the old path. `test/espnet3/systems/test_layout.py` enforces
this layout (`system.py`/`inference.py`/`task.py`/`metrics/`/`src/` are the
only allowed top-level entries in a system package) for every system
going forward.

### 6. `conf_package` for a recipe-local system

`default_conf_package(system_cls)` assumes `system_cls` lives at
`espnet3.systems.<x>.system` and derives its default config package as
`egs3.TEMPLATE.<x>`. A `System` subclass defined inside a recipe itself
(rather than under `espnet3.systems.*`) does not fit that layout, so this
derivation gives the wrong package. Pass `conf_package=` explicitly in
that case:

```python
launch(MyRecipeLocalSystem, conf_package="egs3.my_recipe")
```

### Scope

This note documents only what changed in this PR (the Stages Contract
itself). It does not migrate any other open PR; each PR that adds a
system or recipe affected by the above should migrate itself.
