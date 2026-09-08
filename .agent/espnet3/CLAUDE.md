# `espnet3/` -- package map

See [`.agent/CLAUDE.md`](../CLAUDE.md) for cross-cutting guidance. Each subpackage below has its own
`CLAUDE.md` with the detailed package reference -- this file is just the full directory tree plus
links.

```
espnet3/
├── cli/                        # -> cli/CLAUDE.md
│   ├── main.py                 # argparse dispatcher, registers subcommands
│   └── clone/                  # `espnet3 clone` subcommand
│       ├── command.py          # add_arguments() / run() -- copies + rewrites publication/demo config
│       └── resolver.py         # resolve_recipe() / list_recipes() -- <dataset>/<task> -> egs3/ path
├── systems/                     # -> systems/CLAUDE.md -- one class per "system" (task family)
│   ├── base/                    # BaseSystem + the stage implementations shared by every system
│   │   ├── system.py            # BaseSystem: stage stubs, stage_log_mapping, create_dataset()
│   │   ├── training.py          # collect_stats() / train() free functions used by BaseSystem
│   │   ├── inference.py         # infer(): builds datasets/providers and drives InferenceRunner
│   │   ├── inference_provider.py, inference_runner.py   # infer-stage Runner/Provider pair
│   │   └── metric.py            # measure(): the measure stage
│   ├── asr/                      # ASRSystem and everything ASR-specific
│   │   ├── system.py             # ASRSystem(BaseSystem): train_tokenizer(), tokenizer wiring
│   │   ├── task.py, transducer_task.py   # ASRTask / ASRTransducerTask(AbsTask) -- bridge to espnet2 models
│   │   ├── tokenizers/sentencepiece.py   # prepare_sentences / train_sentencepiece / add_special_tokens
│   │   └── metrics/{cer,ter,wer}.py      # CER / TER / WER(BaseMetric)
│   └── tts/                      # TTSSystem and everything TTS-specific
│       ├── system.py                       # TTSSystem(BaseSystem)
│       └── remove_long_short_{provider,runner}.py   # the remove_long_short filtering stage
├── components/                  # -> components/CLAUDE.md -- building blocks (not stage-shaped)
│   ├── data/                     # dataset abstraction, sharding, collect_stats
│   │   ├── data_organizer.py     # DataOrganizer, DatasetConfig
│   │   ├── dataset.py            # CombinedDataset, DatasetWithTransform, ShardedDataset(ABC)
│   │   ├── dataset_builder.py    # DatasetBuilder(ABC) -- contract every recipe's dataset/builder.py implements
│   │   ├── dataset_module.py     # data_src resolution: load_dataset_module, resolve_dataset_module_name, ...
│   │   ├── dataloader.py         # DataLoaderBuilder
│   │   ├── iterator.py           # EpochSyncIterator
│   │   └── collect_stats.py      # collect_stats_batch / collect_stats, CollectStatsRunner + Provider
│   ├── modeling/
│   │   ├── lightning_module.py   # ESPnetLightningModule -- training_step/validation_step, both optimizer paths
│   │   └── optimization_spec.py  # OptimizerSpec, SchedulerSpec, OptimizationStep, OptimizerRuntimeState
│   ├── trainers/trainer.py       # ESPnet3LightningTrainer -- builds lightning.Trainer from the `trainer:` config
│   ├── callbacks/
│   │   ├── default_callbacks.py  # AverageCheckpointsCallback, MetricsLogger, get_default_callbacks()
│   │   ├── ema.py                # EMACallback
│   │   └── vendored_ema.py       # EMA module (vendored from lucidrains/ema-pytorch)
│   ├── metrics/base_metric.py    # BaseMetric(ABC) -- the contract asr's CER/WER/TER implement
│   └── optimizers/                # placeholder package today -- no custom classes yet; optimizers are wired
│                                   # directly via Hydra `_target_: torch.optim.*` in training.yaml
├── parallel/                     # -> parallel/CLAUDE.md -- Runner/Provider execution model
│   ├── base_runner.py            # BaseRunner(ABC) -- shard planning, locking, dispatch
│   ├── env_provider.py           # EnvironmentProvider(ABC) -- "what one worker needs to run one item"
│   ├── inference_provider.py     # a second, standalone InferenceProvider (distinct from systems/base's)
│   └── parallel.py               # set_parallel/get_parallel_config (module-global context), build_client/get_client (Dask)
├── publication/                  # -> publication/CLAUDE.md -- packaging a trained model / demo
│   ├── inference_model.py        # InferenceModel -- loads a packed bundle (local dir or HF hub tag)
│   └── demo/
│       ├── packing.py            # pack_demo() / upload_demo()
│       ├── assets.py             # UIAsset(base) / UIAssetRegistry / DefaultAudioUI / DefaultTextUI
│       └── session.py            # DemoSession / load_demo_session -- runtime wrapper used by the Gradio app
└── utils/                        # -> utils/CLAUDE.md -- cross-cutting helpers, NOT stage- or system-specific
    ├── config_utils.py           # load_config_with_defaults / load_and_merge_config -- the Hydra/OmegaConf loading layer
    ├── run_utils.py               # apply_training_experiment_context / validate_experiment_context -- bridges the
    │                               # five per-recipe config files (exp_dir/exp_tag/inference_dir propagation)
    ├── stages_utils.py            # resolve_stages / run_stages / parse_cli_and_stage_args -- the --stages CLI loop
    ├── task_utils.py               # get_task_class / get_espnet_model / save_espnet_config -- bridge to espnet2 tasks
    ├── logging_utils.py            # configure_logging / set_stage_log_handler / log_run_metadata / log_component
    ├── publication_utils.py        # pack_model / upload_model + README/meta.yaml generation
    ├── writer_utils.py             # write_artifact -- typed output writer used by the infer stage
    ├── scp_utils.py                 # load_scp_paths, get_class_path
    └── download_utils.py            # setup_logger / download_url / extract_targz / DownloadProgress
```

- [`cli/CLAUDE.md`](cli/CLAUDE.md)
- [`systems/CLAUDE.md`](systems/CLAUDE.md)
- [`components/CLAUDE.md`](components/CLAUDE.md)
- [`parallel/CLAUDE.md`](parallel/CLAUDE.md)
- [`publication/CLAUDE.md`](publication/CLAUDE.md)
- [`utils/CLAUDE.md`](utils/CLAUDE.md)
