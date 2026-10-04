# `espnet3/systems/`

See [`.agent/CLAUDE.md`](../../CLAUDE.md) for cross-cutting guidance.

```
espnet3/systems/
├── base/                    # BaseSystem + the stage implementations shared by every system
│   ├── system.py            # BaseSystem: stage stubs, stage_log_mapping, create_dataset()
│   ├── training.py          # collect_stats() / train() free functions used by BaseSystem
│   ├── inference.py         # infer(): builds datasets/providers and drives InferenceRunner
│   ├── inference_provider.py, inference_runner.py   # infer-stage Runner/Provider pair
│   └── metric.py            # measure(): the measure stage
├── asr/                      # ASRSystem and everything ASR-specific
│   ├── system.py             # ASRSystem(BaseSystem): train_tokenizer(), tokenizer wiring
│   ├── task.py, transducer_task.py   # compatibility copies of espnet2/tasks task implementations
│   ├── tokenizers/sentencepiece.py   # prepare_sentences / train_sentencepiece / add_special_tokens
│   └── metrics/{cer,ter,wer}.py      # CER / TER / WER(BaseMetric)
└── tts/                      # TTSSystem and everything TTS-specific
    ├── system.py                       # TTSSystem(BaseSystem)
    └── remove_long_short_{provider,runner}.py   # the remove_long_short filtering stage
```

A "system" is the class that owns a task family's staged pipeline
(`create_dataset -> train_tokenizer -> collect_stats -> train -> infer -> measure -> pack_model ->
upload_model -> pack_demo -> upload_demo`). Every system subclasses `BaseSystem`.

Keep each system self-contained. Reuse behavior from another system only when it is truly a shared
abstraction; move that behavior into `systems/base`, `components`, `parallel`, or `utils` instead of
making one task family depend on another task family's implementation.

## System directory names

Apply these rules in priority order when naming a directory under `espnet3/systems/`:

1. **Use lowercase consistently.** Every system directory name is lowercase.
2. **Make the name clear and unambiguous.** It must identify the system without being readily
   confused with another system.
3. **Keep it short.** Choose the shortest name that remains clear and unambiguous.
   For system directories, use an established, readily understood abbreviation when it keeps an
   otherwise unwieldy name short. This exception is specific to system names: `esp2`, `asr`, `tts`,
   `st`, and `enh` are accepted terms here. Do not expand `esp2_asr_transducer` to
   `espnet2_asr_transducer` merely to avoid the abbreviation.
4. **Prefer one word.** Do not use an underscore when a clear one-word name is available. Examples:
   `f5tts`, `parakeet`, `granitespeech`, and `speechlm`.
5. **Use underscores only to preserve meaningful structure.** They are appropriate when they are
   part of the established name or distinguish a meaningful version or variant. Examples:
   `owsm_v4`, `owsm_v3`, `qwen2_audio`, and `qwen2_5_omni`.
6. **Name ESPnet2 task-derived systems `esp2_<task>`.** Use `esp2_asr`, `esp2_tts`, `esp2_st`, or
   `esp2_enh` when the system follows the ESPnet2 task structure. The `esp2_` prefix makes
   provenance visible and groups these directories together alphabetically.
7. **Treat other ESPnet2-derived systems as regular systems.** A system without an ESPnet2
   task-based structure keeps its ordinary name, for example `speechlm`. Reconsider the name only
   if that system is later split into distinct systems such as `opus` and `bugpiper`.

The naming practices in the Hugging Face Transformers `models/` directory are a useful reference,
but ESPnet3's conventions and maintenance needs take precedence.

- **`base/system.py`** -- `BaseSystem`: stores the five per-stage configs (`training_config`,
  `inference_config`, `metrics_config`, `publication_config`, `demo_config`), resolves
  `stage_log_mapping` (which directory each stage's log file goes in), and implements
  `create_dataset()` (walks `training_config.dataset.{train,valid,test}`, resolves each entry's
  `data_src` via `dataset_module.load_dataset_module`, and calls the recipe's `DatasetBuilder`).
  `train`/`infer`/`measure`/`pack_model`/`upload_model`/`pack_demo`/`upload_demo` delegate to the
  free functions below.
- **`base/training.py`** -- `collect_stats(config)` / `train(config)`: builds the
  `ESPnetLightningModule` + `ESPnet3LightningTrainer` (see
  [`.agent/espnet3/components/CLAUDE.md`](../components/CLAUDE.md)) and runs stats collection or
  `trainer.fit`.
- **`base/inference.py`**, **`inference_provider.py`**, **`inference_runner.py`** -- the `infer` stage.
  `infer(config)` resolves the configured test sets and dispatches an `InferenceRunner` (a
  `parallel.BaseRunner` subclass -- see
  [`.agent/espnet3/parallel/CLAUDE.md`](../parallel/CLAUDE.md)) backed by an `InferenceProvider` (an
  `EnvironmentProvider`) to produce per-utterance hypotheses via `writer_utils.write_artifact`.
- **`base/metric.py`** -- `measure(metrics_config)`: the `measure` stage; resolves test sets from
  `inference_dir`, instantiates each configured `BaseMetric`, and writes `metrics.json`.
- **`asr/system.py`** -- `ASRSystem(BaseSystem)`: adds the `train_tokenizer` stage
  (`tokenizers/sentencepiece.py`) and wires the trained token list into the model config.
- **`asr/metrics/{cer,ter,wer}.py`** -- metric inputs are not model batches or Dataset samples.
  The `measure` stage reads aligned SCP files from each inference test-set directory and calls a
  metric as `metric(data, test_name, inference_dir)`, where `data` maps aliases such as `ref` and
  `hyp` to `Path` objects. The default metrics expect `ref.scp` and `hyp.scp` with matching
  utterance IDs in the same order; custom metrics should use `BaseMetric.iter_inputs(...)` to enforce
  that alignment. Configure the aliases explicitly when they differ:
  ```yaml
  metrics:
    - metric:
        _target_: my_project.metrics.MyMetric
      inputs:
        reference: ref
        prediction: hyp
  ```
  If `inputs` is omitted, the metric's `ref_key` and `hyp_key` attributes are used. This input
  contract is intentionally different from training/inference data, so take care when adding or
  modifying a metric; its result is written to `${inference_dir}/metrics.json`.
- **`asr/task.py`**, **`asr/transducer_task.py`** -- compatibility copies of the corresponding
  `espnet2/tasks` implementations, used by `utils/task_utils.get_espnet_model` when a recipe sets
  `task:` instead of instantiating `model:` directly via Hydra. These files exist only to preserve
  ESPnet2 backward compatibility: do not edit these files or add new behavior here. `espnet2/tasks`
  is the source of truth; compatibility copies are updated only by the established synchronization
  process when required.
- **`asr/tokenizers/sentencepiece.py`** -- `prepare_sentences`, `train_sentencepiece`,
  `add_special_tokens`: the SentencePiece training pipeline behind `train_tokenizer`.
- **`tts/system.py`** -- `TTSSystem(BaseSystem)`: adds `create_token_list` and the
  `remove_long_short` filtering stage.
- **`tts/remove_long_short_{provider,runner}.py`** -- `RemoveLongShortProvider` /
  `RemoveLongShortRunner`: filters utterances outside a configured duration range, in parallel via
  `parallel.BaseRunner`.

For adding a new stage to a `System`, or subclassing one for recipe-specific behaviour, see the root
guide's "Adding a new pipeline stage" section and [`.agent/egs3/CLAUDE.md`](../../egs3/CLAUDE.md)'s
"When you need a custom System" section.
