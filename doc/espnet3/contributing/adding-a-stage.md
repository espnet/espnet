# Adding a Stage

A system's `stages` tuple declares the stage names, their order, their configuration
roles and their log directories. The recipe entry point delegates to `launch`.

## Start from the existing system

```python
# egs3/mini_an4/esp2_asr/run.py
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ASRSystem)
```

Keep the main guard: worker processes can import the entry point.
There is no `DEFAULT_STAGES` or `parse_cli_and_stage_args` API in the template.
`launch` derives `--stages`, configuration-role flags and directory flags from
`system_cls.stages`; do not duplicate that interface in `run.py`.

## Declare and implement the stage together

For a reusable system, put the entry points in
`espnet3/systems/<name>/system.py`. Keep metrics in `metrics/`, an ESPnet2 task
adapter in `task.py` when needed, and other implementation helpers in `src/`.
Recipe-specific helpers can live under `egs3/<corpus>/<system>/src/`.
Use absolute imports in both shared and recipe code.

This example adds label preparation after dataset creation:

```python
# espnet3/systems/my_system/system.py
from pathlib import Path

from espnet3.contract.stages import StageSpec
from espnet3.systems.esp2_asr.system import ASRSystem


class MySystem(ASRSystem):
    # [:1] is create_dataset. [1:] starts with train_tokenizer, then
    # collect_stats, train, infer, measure and publication/demo stages.
    stages = (
        ASRSystem.stages[:1]
        + (StageSpec(name="prepare_labels", config="training", log_dir="data_dir"),)
        + ASRSystem.stages[1:]
    )

    def prepare_labels(self):
        """Write recipe labels after dataset preparation."""
        config = self.stage_configs["prepare_labels"]
        output = Path(config.prepare_labels.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("example label\n", encoding="utf-8")
```

`StageSpec` accepts keyword arguments. `name` names the callable method;
`config` names a role such as `training` or a tuple of roles;
`log_dir` names a dotted key in the merged stage config, not an instance attribute.
A stage with no `log_dir` uses the experiment directory as its log fallback.

Subclasses inherit their parent's entire tuple unless they assign a replacement.
To add a stage, construct the complete desired tuple as above; declaring only the
new `StageSpec` replaces the inherited tuple rather than extending it.
To override an existing stage's behavior, override its method; retain its spec
unless its role or log directory must also change.

The contract is checked when the subclass is defined:

- stages must be a nonempty tuple with distinct names;
- every declared name must resolve to a callable method;
- every public callable newly defined on the class must be a declared stage.

Name non-stage helpers with a leading underscore. Stage methods take no CLI
arguments; read settings from `self.stage_configs[stage_name]`.

## Put settings in the role's YAML

For the example, add this to the recipe's `conf/training.yaml`:

```yaml
prepare_labels:
  output_path: ${data_dir}/labels.txt
```

Then use the shared launcher:

```python
# egs3/my_corpus/my_system/run.py
from espnet3.systems.launch import launch
from espnet3.systems.my_system.system import MySystem

if __name__ == "__main__":
    launch(MySystem, conf_package="egs3.TEMPLATE.esp2_asr")
```

The explicit `conf_package` reuses ASR defaults for this example. Without it,
`launch` derives a template from the system class's own module, here
`egs3.TEMPLATE.my_system`. A production system can supply its own template.

```bash
python run.py --stages create_dataset prepare_labels --training_config conf/training.yaml
```

`resolve_stages` selects stages in the declared tuple's order, irrespective of
CLI argument order. It does not automatically execute prerequisites. `all` selects
all declared stages, including publication and uploads, so select stages explicitly
while developing.

For a genuinely separate role, declare e.g.
`StageSpec(name="export", config="export", log_dir="export.out_dir")`, implement
`export(self)`, and supply `conf/export.yaml` in the selected template package as
well as the recipe override. The launcher creates `--export_config` automatically.
Without `--exp_dir`, every role of every requested stage must be provided. With
`--exp_dir`, saved-config recovery is attempted; missing or unresolved settings
can still fail during loading or execution.

## Configuration and logging belong to stages

The constructor takes `configs={"training": cfg, "inference": cfg, ...}`,
`exp_dir` and optionally `stages_to_run`. The old `train_config`, `infer_config`,
`stage_log_mapping` and `self.stage_log_dirs` interface is not used.
Each stage has its own `DictConfig`; mutating one stage's config does not mutate
another stage's config object.

`run_stages` calls the method with no arguments and, for each selected stage:

1. Bakes that stage's own keys to `<exp_dir>/config/<stage>.yaml` on rank zero.
2. Writes a full resolved snapshot to `<log_dir>/<stage>.yaml` on rank zero.
3. Attaches `<stage>.log`, records configuration/environment metadata and timing,
   then calls the stage method.

Existing logs are rotated. `--write_requirements` additionally records installed
requirements beside stage logs. `--dry_run` skips stage methods and stage log /
snapshot writing, but parsing, config resolution and system construction still run;
the constructor can create `exp_dir`.

The current launcher keeps only requested stages in `system.stage_configs`.
ASR `train` reads `train_tokenizer`'s config, and `measure` reads `infer`'s config.
Use `--stages train_tokenizer train` or `--stages infer measure` for those flows.
Loading saved configs with `--exp_dir` does not populate omitted stage entries.

The current runner does not synchronize non-training stage bodies across ranks.
Only baking/snapshot writes are rank-gated. Launch preparation once before
starting a distributed training job rather than relying on the stage runner to
prevent duplicate preparation.

## Keep data, model and metric contracts separate

`StageSpec` describes orchestration. It does not declare a dataset's fields,
an inference model's inputs/outputs or a metric's inputs/outputs.

| Boundary | Declaration | Where it belongs |
| --- | --- | --- |
| Recipe dataset | `fields = (Field(...), ...)` | The `BaseDataset` subclass |
| Inference | `contract.inputs` / `contract.outputs` | `conf/inference.yaml`, consumed by the configured runner |
| Metric | `inputs` / `outputs` | The `BaseMetric` subclass, or instance before `super().__init__()` |

For example, put these declarations in a dataset class, not at module scope:

```python
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field


class Dataset(BaseDataset):
    # Field's first argument is the item key; the second is its registered kind.
    fields = (Field("utt_id", "text"), Field("text", "text"))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"utt_id": "u1", "text": "hello"}
```

The dataset contract checks the first retrieved item per instance. Inference's
YAML declaration is separate from these class attributes:

```yaml
input_key: speech
contract:
  inputs:
    speech:
      kind: audio
      sample_rate: 16000
  outputs:
    text:
      kind: text
```

`input_key` must match the contract's input names. Audio with a known rate is
resampled by default, while an unannotated array is assumed to have the declared
rate. The checked value is an `Audio` object. `ASRInferenceRunner` unwraps it for
ESPnet2 and formats n-best output as `text`; a backend already accepting checked
inputs and returning named outputs can use the default `InferenceRunner`.
The configured runner's `forward` path is shared by recipe inference,
`InferenceModel` and the default demo. `output_fn` cannot be combined with a
contract; adapt backend results with the runner's `format_output` hook instead.
A bare `Field("speech", "audio")` preserves channels by default; request channel
selection/averaging with an explicit normalizer when required.

A metric's outputs must all have kind `number`; bind its declared inputs under each
`metrics:` list entry in `conf/metrics.yaml`, for example:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
```

`hyp: text` reads `text.scp`; `ref: dataset:text` reads the original dataset column.
The metric contract is independent of the inference model contract.

## Validate the extension

Check `python run.py --help` and `--dry_run` with the actual required role configs.
Test the new stage method, invalid declarations and missing config. Add integration
coverage for new end-to-end behavior, using `coverage run --append` in CI.
Keep tests parallel to their source location under `test/espnet3/`.

Update the recipe, stage and configuration documentation when the interface changes.

<DocCards :cols="2">
  <DocCard title="CI and PR" href="./ci-and-pr.html" />
  <DocCard title="Docstring guide" href="./docstring-guide.html" />
</DocCards>
