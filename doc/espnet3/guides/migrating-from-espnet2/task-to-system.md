# Task to system

One of the biggest architectural changes in ESPnet3 is the move from `Task`
to `System`.

This change did not happen only for naming reasons.
It happened because the old task-centric abstraction was becoming too narrow.

## Why ESPnet3 moved beyond Task

In ESPnet2, a `Task` class is a strong organizing unit.

That works well when one recipe is mostly:

- one task
- one model family
- one training/inference interface

But that assumption becomes weaker when the system starts to include:

- multiple modalities
- multiple generation styles
- multiple inference paths
- multiple task families inside one larger foundation model

This is especially visible for newer model families such as:

- SpeechLM-style setups
- systems that combine ASR, TTS, SDS, or other capabilities
- future foundation-model recipes that naturally contain several task-like
  subproblems

So the design pressure changed.

ESPnet no longer only needed:

- "a task class that knows how to build one model"

It also needed:

- "a larger owner that can contain tasks, stages, configs, preparation, inference, evaluation, and publication behavior"

That larger owner is `System`.

## The key idea

You can think of the change like this:

- ESPnet2 `Task` is mainly a model-and-data contract owner
- ESPnet3 `System` is a pipeline owner

A `System` can still reuse task-like logic internally.
But it is not limited to being only a task wrapper.

That makes it a better fit for future recipes where one foundation model may
span several older task boundaries.

## What Task usually owns in ESPnet2

An ESPnet2 `Task` class usually owns things like:

- parser / CLI argument definitions
- model construction
- optimizer construction
- collate function construction
- preprocessor construction
- dataset construction rules
- iterator / dataloader conventions
- train / inference entrypoint assumptions

So `Task` is deeply tied to:

- one model family
- one expected batch format
- one training/inference contract

That is powerful, but also restrictive.

## What System owns in ESPnet3

In ESPnet3, `System` owns named stages and the overall recipe workflow.

At the base level, `BaseSystem` owns stage methods such as:

- `create_dataset()`
- `collect_stats()`
- `train()`
- `infer()`
- `measure()`
- `pack_model()`
- `upload_model()`
- `pack_demo()`
- `upload_demo()`

An actual system such as `ASRSystem` can add task-specific logic on top, for
example tokenizer training.

So `System` is not only "how to build the model".
It is also:

- how the recipe runs
- how configs are split across stages
- how logs are organized
- how dataset creation is triggered
- how inference and metrics are connected
- how publication and demo stages are handled

## Side-by-side role comparison

| Topic | ESPnet2 `Task` | ESPnet3 `System` |
| --- | --- | --- |
| Main abstraction | task-centric model/data owner | pipeline and stage owner |
| Main unit of execution | task-specific train/inference flow | named stages in `run.py` |
| Config ownership | often one task-centered config surface | split across `training.yaml`, `inference.yaml`, `metrics.yaml`, ... |
| Dataset contract | strongly tied to task expectations | stage-specific, config-driven, can still reuse task logic |
| Publication / demo | often outside the main task abstraction | normal stage methods |
| Multi-task / foundation-model fit | can become stretched | designed to contain broader workflows |


## Task is not completely gone

ESPnet3 does not reject task logic.

In fact, one migration path is:

- keep using an ESPnet2 task class
- let `training.yaml` set `task`
- let the `System` call into that task-side model path

So the relationship is not:

- "Task disappeared"

It is more like:

- "Task became one possible internal implementation detail, while System became the outer owner"

That is an important distinction.

## Practical migration view

When porting from ESPnet2, use this mapping:

| ESPnet2 idea | ESPnet3 idea |
| --- | --- |
| task class | system class |
| task entrypoint | stage method |
| one large task config | multiple stage-oriented configs |
| task-specific shell stages | `System` stage list in `run.py` |
| decode/scoring shell flow | `infer()` and `measure()` |

## What changes most for users

If you were used to ESPnet2, the biggest user-facing changes are:

1. you think in stages, not stage numbers
2. you pass several config files, not one task-centered config
3. inference and metrics become normal config-driven stages
4. publication and demo also become stage-driven

So even when the underlying model still resembles an ESPnet2 task-based model,
the top-level workflow feels different.


## Compare the responsibilities directly

### ESPnet2 Task responsibilities

Typical responsibilities:

- define arguments
- define model construction
- define preprocessor and collate behavior
- define iterator and dataset expectations
- define task-specific training and inference assumptions

### ESPnet3 System responsibilities

Typical responsibilities:

- own stage methods
- connect stage names to config files
- coordinate dataset creation, stats, training, inference, metrics, publication
- provide one recipe-level workflow owner
- optionally reuse task-side model construction where useful

So the level of abstraction is simply different.

`Task` is closer to the model/data contract.
`System` is closer to the experiment workflow contract.


## Related pages

<DocCards :cols="3">
  <DocCard
    title="System and stages"
    desc="Read the overview of `run.py`, stage dispatch, and config mapping."
    icon="tabler:hierarchy-2"
    href="../../stages/index.html"
  />
  <DocCard
    title="Stages"
    desc="See the named stage map used by ESPnet3 recipes."
    icon="tabler:route"
    href="../../stages/index.html"
  />
  <DocCard
    title="Config diff"
    desc="See how task-centered ESPnet2 config surfaces split into ESPnet3 files."
    icon="tabler:settings-2"
    href="./config-diff.html"
  />
  <DocCard
    title="Cluster and parallel"
    desc="See how shell-driven task execution changes in ESPnet3."
    icon="tabler:binary-tree-2"
    href="./cluster-and-parallel.html"
  />
</DocCards>
