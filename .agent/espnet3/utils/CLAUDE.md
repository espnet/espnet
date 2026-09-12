# `espnet3/utils/`

See [`.agent/CLAUDE.md`](../../CLAUDE.md) for cross-cutting guidance.

```
espnet3/utils/
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

Cross-cutting helpers used across systems/components/parallel. If a helper here starts encoding
stage- or system-specific behaviour, it likely belongs in `systems/` or `components/` instead (see
[`.agent/espnet3/systems/CLAUDE.md`](../systems/CLAUDE.md) /
[`.agent/espnet3/components/CLAUDE.md`](../components/CLAUDE.md)).

- **`config_utils.py`** -- `load_config_with_defaults`, `load_and_merge_config`,
  `load_default_config`: the Hydra/OmegaConf config-loading layer every `run.py` calls. Also defines
  the custom resolvers `self_name` (derives `exp_tag` from the loaded config's filename) and
  `config_path`, and `set_corpus_and_system`.
- **`run_utils.py`** -- `apply_training_experiment_context`, `validate_experiment_context`,
  `resolve_loaded_configs`: bridges the five independently-loaded per-recipe config files, copying
  identity fields (`exp_tag`, `exp_dir`, `inference_dir`, ...) from `training_config`/
  `inference_config` into the others so `measure`/`publication`/`demo` configs can reference them.
- **`stages_utils.py`** -- `resolve_stages`, `run_stages`, `parse_cli_and_stage_args`: parses
  `--stages`, resolves `all`/aliases against a recipe's stage list, and runs each requested stage
  method on the `System` in canonical order.
- **`task_utils.py`** -- `get_task_class`, `get_espnet_model`, `save_espnet_config`: resolves a
  dotted `espnet2.tasks.*` task path and builds/serializes an `AbsESPnetModel` from a `model:` block
  that sets `task:`.
- **`logging_utils.py`** -- `configure_logging`, `set_stage_log_handler`, `log_stage`/
  `log_stage_metadata`, `log_run_metadata`, `log_env_metadata`, `log_component`,
  `log_instance_dict`: per-stage log file setup and structured logging of run metadata (git commit,
  `pip freeze`, environment) and of arbitrary component instances for debugging.
- **`publication_utils.py`** -- `pack_model`, `upload_model`: collects config/checkpoint/tokenizer/
  stats artifacts into a bundle with rewritten relative paths, plus the README/`meta.yaml` generation
  helpers. Consumed by [`.agent/espnet3/publication/CLAUDE.md`](../publication/CLAUDE.md)'s
  `InferenceModel`.
- **`writer_utils.py`** -- `write_artifact`: the typed output writer (`scp`/`text`/`npy`/custom
  `_target_`) the `infer` stage uses to persist per-utterance outputs.
- **`scp_utils.py`** -- `load_scp_paths`, `get_class_path`: small SCP-file and class-path helpers
  shared by `metrics/` and `publication_utils.py`.
- **`download_utils.py`** -- `setup_logger`, `download_url`, `extract_targz`, `DownloadProgress`:
  generic download-with-progress + tar/zip extraction helpers, available for a recipe's
  `dataset/builder.py` to use -- as of this writing nothing in the shipped recipes calls them yet, so
  treat them as unproven; test whatever you use here.
