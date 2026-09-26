# Adding a Stage

Add a new stage when the behavior does not fit an existing stage or a config-only change.

In ESPnet3, a stage is a method on a `System` class that is selected from
`run.py` with `--stages`.

## How a stage works

A recipe runner defines the available stage names in `run.py`.
The runner parses `--stages`, resolves the requested names, and then calls the
matching methods on the `System` instance.

In practice, this means:

- `run.py` decides which stage names exist
- the `System` class implements the actual methods
- stage settings should live in YAML configs, not ad-hoc CLI flags

This is the core pattern:

```python
from egs3.TEMPLATE.asr.run import DEFAULT_STAGES, build_parser, main, parse_cli_and_stage_args
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=ASRSystem, stages=stages_to_run)
```

`egs3/TEMPLATE/asr/run.py` only exports `DEFAULT_STAGES` (there is no
`ALL_STAGES` in the template); a recipe that adds stages defines its own
`ALL_STAGES` list locally (Step 3 below).

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

## Step 2: implement the method on the System

Add the stage as a method on your system class.

In practice, this is also how you handle a custom system.
If the new stage is recipe-specific, create a recipe-local system file and put
the stage method there.
If the stage should be shared across many recipes of the same system, extend the
shared system under `espnet3/systems/<system>/system.py`.

Typical locations are:

- recipe-local system: `egs3/<recipe>/<system>/src/system.py`
- shared ESPnet3 system: `espnet3/systems/<system>/system.py`

```python
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    def prepare_labels(self):
        ...
```

If `run.py` allows `--stages prepare_labels`, the system must implement
`prepare_labels(self)`.

Keep the method small and stage-focused.
If the logic is large, split and move helper code into modules.


## Step 3: expose it from run.py

Add the new stage name to the ordered stage list in your recipe runner.

```python
from egs3.TEMPLATE.asr.run import DEFAULT_STAGES

ALL_STAGES = [
    *DEFAULT_STAGES,
    "prepare_labels",
]
```

The stage order in this list is the canonical order.
Even if a user writes `--stages infer train`, the resolved execution order
follows the order defined in `ALL_STAGES`.

::: note
Stage order is controlled by the runner, not by the CLI input order.
Keep the canonical order in `run.py` stable and easy to read.
:::

## Step 4: wire the config it needs

Stages should not take custom CLI arguments directly.
Put their settings in YAML and load the config in `run.py`.

If your stage can reuse an existing config such as `training.yaml`, prefer that.
If it needs a separate config, add a new `--*_config` flag and load it there.

Typical pattern:

```python
parser.add_argument(
    "--export_config",
    default=None,
    type=Path,
    help="Hydra config for export-related stages.",
)
```

Then load and pass it into the system:

```python
export_config = (
    None
    if args.export_config is None
    else load_config_with_defaults(args.export_config)
)

system = system_cls(
    train_config=train_config,
    infer_config=infer_config,
    metric_config=metric_config,
    export_config=export_config,
)
```

And store it on the system:

```python
class MySystem(BaseSystem):
    def __init__(self, export_config=None, **kwargs):
        super().__init__(**kwargs)
        self.export_config = export_config
```

## Step 5: fail early when config is missing

Config requirements are usually enforced in `run.py`, not inferred from the
system automatically.

Add your stage to the stage-to-config checks:

```python
required_configs = {
    "train": train_config,
    "infer": infer_config,
    "measure": metric_config,
    "export": export_config,
}
```

If a stage requires a config, raise before stage execution starts.
Failing early makes recipe behavior easier to debug.

## Step 6: keep logging and outputs consistent

ESPnet3 runs stages through `espnet3.utils.stages_utils.run_stages()`.
That helper:

- resolves stage order
- attaches per-stage logs
- records timing
- calls the matching system method

The runner entrypoint should also use
`espnet3.utils.logging_utils.configure_logging()` so console and file logging
follow the same format as other ESPnet3 runs.

Per-stage metadata is written through
`espnet3.utils.logging_utils.log_stage_metadata()`.
That keeps the CLI arguments, config paths, environment metadata, and resolved
config contents visible in stage logs.

Stage-level log destinations are resolved on the `System` side.
`espnet3.systems.base.system.BaseSystem` builds `self.stage_log_dirs` from the
standard config paths, and `run_stages()` uses that mapping when it attaches
per-stage file logs.

Typical runner usage:

```python
from espnet3.utils.logging_utils import configure_logging, log_stage_metadata
from espnet3.utils.stages_utils import run_stages


logger = configure_logging()

logger.info("System: %s", system_cls.__name__)
logger.info("Requested stages: %s", args.stages)
logger.info("Resolved stages: %s", stages_to_run)

log_stage_metadata(logger, system=system, args=args)

run_stages(
    system=system,
    stages_to_run=stages_to_run,
    args=args,
    log=logger,
)
```

In standard runners, `run_stages()` already calls `log_stage_metadata()` for
each stage log.

Typical `System`-side setup:

```python
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    def __init__(self, prepare_labels_config=None, **kwargs):
        super().__init__(
            stage_log_mapping={
                "prepare_labels": "training_config.exp_dir",
            },
            **kwargs,
        )
        self.prepare_labels_config = prepare_labels_config
```

Use `stage_log_mapping` when a new stage should write logs under a specific
recipe directory instead of the default fallback.
The resolved directory receives a stage log named `<stage>.log`.
If that file already exists, ESPnet3 rotates it automatically to names such as
`<stage>1.log`, then writes the new run to a fresh `<stage>.log`.
So `<stage>.log` is always the latest stage log.

Try to keep new stages consistent with the existing pipeline:

- write outputs into the recipe directory layout
- use the expected `data/`, `exp/`, and `logs/` locations
- log enough information to reproduce the run

If your stage introduces a new config, include it in stage metadata logging in
the same way as the standard configs.

## Step 7: add tests

New stages should be tested.

At minimum:

- add unit tests for the new behavior
- add runner-level tests if `run.py` wiring changed
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
egs3/my_corpus/asr/
├── run.py
├── conf/
│   └── training.yaml
└── src/
    └── my_system.py
```

Example `src/my_system.py`:

```python
from espnet3.systems.base.system import BaseSystem


class MySystem(BaseSystem):
    def __init__(self, prepare_labels_config=None, **kwargs):
        super().__init__(
            stage_log_mapping={
                "prepare_labels": "training_config.exp_dir",
            },
            **kwargs,
        )
        # Keep the custom stage config on the system instance.
        self.prepare_labels_config = prepare_labels_config

    def prepare_labels(self):
        print("Preparing labels")
        # Add your stage logic here.
```

Example `run.py`:

```python
from pathlib import Path

from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.utils.config_utils import load_config_with_defaults

from src.my_system import MySystem

ALL_STAGES = [
    *DEFAULT_STAGES,
    # Extend the template stage list with one custom stage.
    "prepare_labels",
]

if __name__ == "__main__":
    # Reuse the template parser and add only recipe-specific arguments.
    parser = build_parser(stages=ALL_STAGES)
    parser.add_argument(
        "--prepare_labels_config",
        type=Path,
        default=None,
        help="Config for the prepare_labels stage.",
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=ALL_STAGES)

    prepare_labels_config = (
        None
        if args.prepare_labels_config is None
        else load_config_with_defaults(args.prepare_labels_config)
    )

    # Template main() validates the standard stages; guard only the custom one here.
    if "prepare_labels" in stages_to_run and prepare_labels_config is None:
        raise ValueError(
            "prepare_labels stage requires --prepare_labels_config."
        )

    # Inject the custom config while keeping the template main() flow.
    main(
        args=args,
        system_cls=lambda **kwargs: MySystem(
            prepare_labels_config=prepare_labels_config,
            **kwargs,
        ),
        stages=ALL_STAGES,
    )
```

You can then run only the custom stage with:

```bash
python run.py --stages prepare_labels --prepare_labels_config conf/prepare_labels.yaml
```

## Common mistakes

- putting stage settings in CLI flags instead of YAML
- forgetting to add the stage name to `run.py`
- forgetting to enforce the required config
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
