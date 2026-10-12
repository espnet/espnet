# Speech Language Model

This template contains two drivers for different SpeechLM training interfaces:

| Driver | Python entry point | Use |
| --- | --- | --- |
| `speechlm.sh` | `espnet2.bin.speechlm_train` | Existing staged recipes, such as LibriTTS and mini_an4, with the traditional ESPnet environment and job launcher |
| `train.sh` | `espnet2.speechlm.bin.train` | Bagpiper and Bagpiper-TTS training with TorchTitan, prepared dataset manifests, and length statistics |
| `stage_utils.sh` | sourced by those recipes' `run.sh` | Finding the latest complete checkpoint, and the rule for where a training stage starts |

The drivers have different configuration and data interfaces. Keep using
`speechlm.sh` for recipes built around it; `train.sh` provides the entry point for
the current SpeechLM trainer. The existing `setup.sh` scaffolds the traditional
`speechlm.sh` layout.

## Recipes without data preparation

[Bagpiper](../../bagpiper/speechlm1/README.md) and
[Bagpiper-TTS](../../bagpiper_tts/speechlm1/README.md) contain `run.sh`, `conf/`, and a
README. Each `run.sh` enters its recipe directory and writes its stages out in
the usual egs2 shape, ending in export and inference:

| | [Bagpiper](../../bagpiper/speechlm1/run.sh) | [Bagpiper-TTS](../../bagpiper_tts/speechlm1/run.sh) |
| --- | --- | --- |
| 1 | warmup | SFT |
| 2 | pretraining | export |
| 3 | SFT | inference |
| 4 | export | |
| 5 | inference | |

`./run.sh` runs them in order and `--stage` / `--stop-stage` select part of
that, so the last stage decodes published weights without training anything
when it is given `--export-path` and `--train-config`. A training stage
continues its own output directory once that has checkpoints, and otherwise
starts from the stage before it; `stage_utils.sh` holds that rule and the
search for the latest complete checkpoint, so the recipes do not repeat it.

Their inputs are already prepared, so the data-preparation and cluster-launch
files are unnecessary:

- **Environment:** there is no `path.sh` to activate `tools/venv`. Activate the
  environment from the [SpeechLM installation guide](../../../espnet2/speechlm/INSTALL.md)
  first, or pass `--python /path/to/env/bin/python`. `train.sh` adds the repository
  root to `PYTHONPATH` and uses that Python interpreter.
- **Launch:** there is no `cmd.sh`, `${cuda_cmd}`, `run.pl`, or `slurm.pl` dispatch.
  `train.sh` invokes `torchrun` directly. For multiple nodes, start the command on
  every allocated node with the same `--num-nodes`, `--master-addr`, and
  `--master-port`, and a distinct `--node-rank`. `--ngpu` means GPUs per node;
  allocate nodes with your scheduler before launching.
- **Data:** there is no `local/data.sh` or `db.sh`, and the stages above do not
  prepare anything. Supply prepared manifests and length statistics directly. The `local/`, `scripts`,
  `pyscripts`, `steps`, and cluster configuration files used by the traditional
  pipeline are therefore omitted.

The default configuration is `conf/train.yaml`; additional stages live in
`conf/tuning/`. See each recipe's README for stage selection and checkpoint usage.
