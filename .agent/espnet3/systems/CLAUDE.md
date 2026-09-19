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
