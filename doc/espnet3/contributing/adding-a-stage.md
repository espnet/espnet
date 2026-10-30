# Adding a Stage

Add a new stage when the behavior does not fit an existing stage or a config-only change.

In ESPnet3, a stage is a method on a `System` class, declared in that class's
`stages` tuple of `StageSpec`s. `espnet3.systems.launch.launch()` builds the
CLI from `stages`, resolves which stage methods were requested, and runs them
in declared order — there is no per-recipe CLI/dispatch code to maintain in
`run.py`.

## How a stage works

A `System` subclass declares its stages as a class attribute:

```python
from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
    )
```

In practice, this means:

- the `System` class owns both the stage list and the stage methods
- `--stages` and one `--<role>_config` flag per distinct config role are
  generated from `stages`, not hand-written
- stage settings live in YAML configs, read from `self.stage_configs["<name>"]`
  inside the method

This is the core pattern for a recipe's `run.py`:

```python
from espnet3.systems.launch import launch

from src.my_system import MySystem

if __name__ == "__main__":
    launch(MySystem)
```

`launch()` builds the parser from `MySystem.stages`, resolves which stages
were requested (`--stages all`, or omitting `--stages`, selects every
declared stage), and runs them in the order declared in `stages` — regardless
of the order given on the command line.

## Step 1: check whether you really need a new stage

Before adding one, check:

- can this be handled by an existing stage?
- can this be expressed as a config change?
- is this behavior specific to one system or one recipe?

Good examples of custom stages:

- `prepare_labels`
- `dump_features`
- `export_onnx`

Use short verb-style snake_case names.

## Step 2: implement the method and declare its `StageSpec`

Add the stage as a method on your system class, and add a matching
`StageSpec` to `stages` at the position it should run in.

If the new stage is recipe-specific, create a recipe-local system file and
put the stage + its `StageSpec` there. If the stage should be shared across
many recipes of the same system, extend the shared system under
`espnet3/systems/<system>/system.py`.

Typical locations are:

- recipe-local system: `egs3/<recipe>/<system>/src/system.py`
- shared ESPnet3 system: `espnet3/systems/<system>/system.py`

```python
from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    # Insert prepare_labels right after create_dataset, ahead of the
    # inherited collect_stats/train/... stages.
    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="prepare_labels", config="training", log_dir="data_dir"),
        *BaseSystem.stages[1:],
    )

    def prepare_labels(self):
        config = self.stage_configs["prepare_labels"]
        ...
```

If `--stages prepare_labels` is requested, the system must implement
`prepare_labels(self)`; `BaseSystem.__init_subclass__` checks `stages`
against the class's own methods and raises `StageContractError` at class
definition time if one is missing.

Keep the method small and stage-focused.
If the logic is large, split and move helper code into modules.

## Step 3: pick the config role and log directory

`StageSpec.config` names the config role the stage reads
(`self.stage_configs["prepare_labels"]` comes from whichever role you
declare, e.g. `"training"` for the example above), and `StageSpec.log_dir`
is a dotted key inside that role's config that resolves to the stage's log
directory. Reuse an existing role (like `training`) when the stage's
settings naturally belong there; declare a new role only when the stage
needs its own standalone config file (a new `--<role>_config` flag is then
generated automatically).

::: note
Stage order is controlled by the position of each `StageSpec` in `stages`,
not by CLI input order. Keep that order stable and easy to read.
:::

## Step 4: wire the config it needs

Stages should not take custom CLI arguments directly.
Put their settings in YAML; `self.stage_configs["<name>"]` resolves it for
you from whichever role the stage's `StageSpec.config` names.

If your stage can reuse an existing config role such as `training`, prefer
that — no extra CLI flag or constructor plumbing needed. If it needs a
separate config, declare it as its own role in `StageSpec.config` (e.g.
`config="export"`); `launch()` then exposes `--export_config` for it
automatically.

## Step 5: fail early when config is missing

`check_requested_stages` (called by `launch()`) already raises a
`StageContractError` before any stage runs if a requested stage's config
role was not given and there is no `--exp_dir` to inherit it from. Add
stage-specific validation inside the method itself for anything beyond
"was this role's config provided at all" — for example, a required key
within that config:

```python
def prepare_labels(self):
    config = self.stage_configs["prepare_labels"]
    if config.get("label_map") is None:
        raise ValueError("prepare_labels stage requires training_config.label_map")
    ...
```

Failing early makes recipe behavior easier to debug.

## Step 6: keep logging and outputs consistent

ESPnet3 runs stages through `espnet3.utils.stages_utils.run_stages()`,
called internally by `launch()`. That helper:

- resolves stage order from `stages`
- attaches per-stage logs
- records timing
- calls the matching system method

You do not need to call `run_stages()`, `configure_logging()`, or any
per-stage metadata logging yourself when using `launch()` — it wires all of
that for you. Stage-level log destinations come from each `StageSpec`'s
`log_dir`, resolved against `self.stage_configs[<role>]`; if `log_dir` is
`None`, the stage falls back to the system's own default log directory.

If that file already exists, ESPnet3 rotates it automatically to names such
as `<stage>1.log`, then writes the new run to a fresh `<stage>.log`.
So `<stage>.log` is always the latest stage log.

Try to keep new stages consistent with the existing pipeline:

- write outputs into the recipe directory layout
- use the expected `data/`, `exp/`, and `logs/` locations
- log enough information to reproduce the run

## Step 7: add tests

New stages should be tested.

At minimum:

- add unit tests for the new behavior
- add tests for the `StageSpec`/stage-method wiring if you touched it
- add integration tests for new end-to-end behavior when needed

Try to mirror the package layout under `test/`.

For example:

```text
espnet3/systems/foo/system.py
test/espnet3/systems/foo/test_system.py
```

## Step 8: update the docs

If the new stage is user-facing, update the docs.

Usually that means:

- update the recipe docs if the stage is recipe-specific
- update `stages/` docs if the stage introduces reusable behavior
- add or update config docs if a new config file is required

Do not leave required documentation for later if the feature depends on it.

## Minimal recipe example

Here is a small example of a recipe that adds a custom `prepare_labels` stage.

```text
egs3/my_corpus/esp2_asr/
├── run.py
├── conf/
│   └── training.yaml
└── src/
    └── my_system.py
```

Example `src/my_system.py`:

```python
from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.esp2_asr.system import ASRSystem


class MySystem(ASRSystem):
    stages: ClassVar[tuple[StageSpec, ...]] = (
        ASRSystem.stages[0],
        StageSpec(name="prepare_labels", config="training", log_dir="data_dir"),
        *ASRSystem.stages[1:],
    )

    def prepare_labels(self):
        config = self.stage_configs["prepare_labels"]
        if config.get("label_map") is None:
            raise ValueError(
                "prepare_labels stage requires training_config.label_map"
            )
        print("Preparing labels")
        # Add your stage logic here.
```

Example `run.py`:

```python
from espnet3.systems.launch import launch

from src.my_system import MySystem

if __name__ == "__main__":
    launch(MySystem)
```

You can then run only the custom stage with:

```bash
python run.py --stages prepare_labels --training_config conf/training.yaml
```

## Common mistakes

- putting stage settings in CLI flags instead of YAML
- forgetting to add the `StageSpec` for the new stage (or forgetting the
  matching method — `BaseSystem.__init_subclass__` raises `StageContractError`
  for either mismatch)
- relying on CLI order instead of the declared `stages` order
- skipping tests for new stage behavior

## Related docs

<DocCards :cols="3">
  <DocCard
    title="System and stages"
    desc="See how systems, run.py, and stage execution fit together."
    icon="tabler:hierarchy-2"
    href="../stages/index.html"
  />
  <DocCard
    title="Stage reference"
    desc="Read the stage docs for train, inference, metrics, and more."
    icon="tabler:route"
    href="../stages/index.html"
  />
  <DocCard
    title="Config overview"
    desc="See how stage settings are loaded from YAML configs."
    icon="tabler:settings-2"
    href="../config/index.html"
  />
</DocCards>
