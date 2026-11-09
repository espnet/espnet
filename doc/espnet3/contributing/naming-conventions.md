# Naming Conventions

Naming matters a lot in ESPnet3.

Good names make configs easier to read, stage flows easier to follow, and code
search much more reliable for both humans and coding agents.

This page summarizes the naming patterns currently used in the ESPnet3 codebase
and docs.

::: important
Prefer existing names over inventing new ones.
If the repository already has a stable term for a concept, reuse that term
across code, configs, docs, and file paths.
:::

## General rule

A good ESPnet3 name should be:

- consistent with existing code
- easy to search
- specific about its role
- stable across code, config, and docs

If a new name would create a synonym for an existing concept, do not add it.

## System directory names

Apply these rules, in order, to names under `espnet3/systems/`:

1. Use lowercase.
2. Keep the name clear and unambiguous.
3. Use the shortest name that remains clear. For a system directory, established
   domain abbreviations are appropriate when they prevent an unwieldy name:
   `esp2`, `asr`, `tts`, `st`, and `enh` are accepted terms. This is a deliberate
   exception to the general preference for full words in variables, files, and APIs.
4. Prefer one word where possible. Use underscores only when they preserve an
   established name, version, variant, or meaningful structure.

ESPnet2 task-derived systems use `esp2_<task>`, such as `esp2_asr` and
`esp2_asr_transducer`. The short prefix keeps directory names manageable while
grouping their ESPnet2 provenance in alphabetical listings. ESPnet2-derived work
without a task-shaped system remains a regular system name, such as `speechlm`.

<details>
<summary>Examples of system directory names</summary>

| Name | Why it fits |
| --- | --- |
| `f5tts` | Established one-word system name. |
| `openbeats` | Established one-word system name. |
| `parakeet` | Clear one-word system name. |
| `granitespeech` | Clear one-word system name. |
| `speechlm` | Established name; no separator is needed. |
| `whisper` | Established one-word system name. |
| `wav2vec2` | Established one-word system name. |
| `wavlm` | Established one-word system name. |
| `hubert` | Established one-word system name. |
| `seamlessm4t` | Established one-word system name. |
| `owsm_v3` | The underscore preserves a meaningful version. |
| `owsm_v4` | The underscore preserves a meaningful version. |
| `qwen2_audio` | The underscore preserves the established model family structure. |
| `qwen2_5_omni` | The underscores preserve the established version and variant. |
| `esp2_asr` | ESPnet2 ASR task-derived system. |
| `esp2_asr_transducer` | ESPnet2 ASR transducer system; the short `esp2` prefix keeps it manageable. |
| `esp2_tts` | ESPnet2 TTS task-derived system. |
| `esp2_st` | ESPnet2 speech translation task-derived system. |
| `esp2_enh` | ESPnet2 speech enhancement task-derived system. |
| `huggingface_asr` | A toolkit plus task clearly identifies a wrapper system. |

</details>

## Python naming

Follow normal Python naming first:

- functions and variables: `lower_with_under`
- classes: `CapWords`
- constants: `CAPS_WITH_UNDER`
- module files: `lower_with_under.py`

For internal helpers, use a leading underscore:

- `_normalize_stage_name`
- `_load_inference_config`

## Function names should read like actions

In practice, ESPnet3 function names usually start with a verb or an
action-oriented prefix.

Good examples:

- `load_config_with_defaults`
- `resolve_stages`
- `run_stages`
- `collect_stats`
- `prepare_sentences`
- `upload_model`

Avoid vague noun-like function names such as:

- `stage_config`
- `dataset_module`
- `model_state`

If the function does something, the name should sound like that action.

::: note
Common ESPnet3 action prefixes include `load`, `build`, `run`, `collect`,
`validate`, `prepare`, `infer`, `measure`, `pack`, and `upload`.
:::

## Class names should read like entities

Class names should look like nouns, roles, or objects.

Good examples:

- `BaseSystem`
- `InferenceRunner`
- `DatasetBuilder`
- `ESPnetLightningModule`

Avoid class names that read like commands:

- `LoadDataset`
- `RunInference`
- `BuildTrainer`

If the code represents a thing, the name should read like a thing.

## File names should also read like entities

Python file names should also look like nouns or role names, not verb phrases.

Good examples:

- `system.py`
- `inference_runner.py`
- `dataset_builder.py`
- `config_utils.py`

Avoid names like:

- `run_stage.py`
- `load_config.py`
- `prepare_dataset.py`

When a file holds a role or domain concept, name it after that concept.

## Prefer inference, not decode

For new ESPnet3 code, do not introduce `decode` when the concept is inference
output or inference-time behavior.

Historically, `decode` was a natural name for ASR-style pipelines.
But ESPnet3 now covers more generative systems, and many of those workflows are
not best described as "decoding".
`inference` is the broader and more stable term, so new naming should prefer
that direction.

Prefer names such as:

- `inference`
- `inference_dir`
- `inference_config`
- `InferenceProvider`

Avoid new names such as:

- `decode_dir`
- `decode_config`
- `decode_output`

This rule applies to:

- config keys
- path names
- stage-facing field names
- publication settings
- docs and examples

::: warning
Older code may still contain `decode` in some places.
Do not copy that naming into new ESPnet3 work unless you are intentionally
working inside a legacy compatibility boundary.
:::

## Use snake_case for stage names and config keys

Stage names should be short verb-style snake_case names:

- `create_dataset`
- `train_tokenizer`
- `collect_stats`
- `prepare_labels`
- `pack_model`

Config keys and path-related fields should also use snake_case:

- `training_config`
- `inference_dir`
- `output_keys`
- `stage_log_mapping`

Do not mix styles such as camelCase or kebab-case into Python-facing config.


## Imports under egs3/

Do not use relative imports inside `egs3/`.

Use explicit absolute module paths:

```python
from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder
```

Avoid:

```python
from .builder import MiniAn4Builder
```

Absolute imports are easier to search, easier to move, and easier for coding
agents to reason about.

## Name by boundary

Use names that match the boundary where the code lives.

In shared `espnet3/` code:

- prefer reusable, system-level names
- avoid recipe-specific corpus names

In recipe-local `egs3/<recipe>/<system>/` code:

- recipe-specific names are fine
- names should still stay consistent with the shared API

Examples:

- shared: `InferenceRunner`, `DataOrganizer`, `BaseMetric`
- recipe-local: `MiniAn4Builder`, `LibriSpeechDataset`

## Do not create unnecessary synonyms

Naming drift usually starts when two words mean the same thing.

Examples of bad drift:

- `infer` vs `decode`
- `metric` vs `measure` used inconsistently for the same concept
- `dataset_dir` vs `data_root` vs `corpus_dir` for the same path

Before adding a new name, search the repository and reuse the dominant term if
the meaning is the same.

## Avoid unnecessary abbreviations

Prefer full words unless the abbreviated form is already widely understood in
the field or already well established in ESPnet3.

Good:

- `collect_stats`
- `inference_dir`
- `config_utils`

Usually avoid:

- `prep_data`
- `cfg_loader`
- `infer_out`

Prefer:

- `prepare_data`
- `config_loader`
- `inference_output`

Short forms are fine when they are already common and unambiguous.
`stats` is a good example.
Ad-hoc shortening is not.

::: note
If a shortened name saves only a few characters but makes the meaning less
obvious, use the full word.
:::

## A practical naming workflow

Before introducing a new name:

1. Search for the concept in `espnet3/` and `egs3/`.
2. Reuse an existing name if one already matches.
3. Check whether the name matches the expected Python style.
4. Check whether the name is specific to shared code or recipe-local code.
5. Verify the actual file or directory name before writing links or imports.

## Examples

Good:

```python
def load_inference_config(path: Path) -> DictConfig:
    ...


class InferenceProvider:
    ...
```

Good:

```python
stage_log_mapping = {
    "infer": "inference_config.inference_dir",
    "measure": "metrics_config.inference_dir",
}
```

Bad:

```python
def config_loader(path: Path) -> DictConfig:
    ...


class RunInference:
    ...
```

Bad:

```python
decode_dir = "exp/my_run/decode"
from .builder import MyBuilder
```

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Adding a stage"
    desc="See how stage names and custom system methods fit together."
    icon="tabler:route"
    href="./adding-a-stage.html"
  />
  <DocCard
    title="Docstring guide"
    desc="Write documentation that matches these naming choices."
    icon="tabler:file-text"
    href="./docstring-guide.html"
  />
  <DocCard
    title="CI and PR"
    desc="See review expectations for naming and cleanup changes."
    icon="tabler:git-pull-request"
    href="./ci-and-pr.html"
  />
</DocCards>
