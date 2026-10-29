---
title: Getting Started with ESPnet3
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🚀 Getting Started with ESPnet3

This guide provides the fastest way to start using ESPnet3.
Choose the workflow that fits your environment and follow the examples below.


# ⚡ Quick Start (ASR Example)

## 1. Install ESPnet3

Pick a system that is closest to your work.
And then you can install only packages that is required for your work.

<HomeQuickStart />

## 2. Open a recipe from the source checkout

This implementation has no `espnet3 clone` command. Install the checkout with
training dependencies and use a recipe that is actually present:

```bash
git clone https://github.com/espnet/espnet.git
cd espnet
python -m pip install -e ".[asr,train]"
cd egs3/mini_an4/esp2_asr
```

Mini-AN4 is the small smoke-test recipe; use
`egs3/librispeech_100/esp2_asr` or `egs3/spgispeech/esp2_asr` for their respective
corpora. Read each recipe's dataset preparation requirements before running.
The dataset builder converts Mini-AN4 SPH files with `sph2pipe`, which must be
available through the recipe's environment.

A recipe contains `run.py`, configuration overrides in `conf/`, a `dataset/`
package exporting `Dataset` and `DatasetBuilder`, and optional helpers in `src/`.
The `egs3/TEMPLATE/esp2_asr` package supplies shared defaults. Corpus data,
checkpoints and inference outputs are generated artifacts, not bundled models.

To see how these pieces connect, read
[What a recipe is](./what-is-a-recipe.html).

## 3. Use the shared launcher

The recipe's `run.py` delegates to `launch`:

```python
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ASRSystem)
```

The guard prevents multiprocessing workers from launching the recipe again.
`StageSpec` declarations on the system determine stage names, config roles and
stage directories. There is no separate recipe-maintained stage-name list.

## 4. Run the baseline

Run the training prerequisites and inference/measurement in the same request:

```bash
python run.py \
  --stages create_dataset train_tokenizer collect_stats train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Use one of the recipe's concrete training overrides if it does not provide
`conf/training.yaml`; Mini-AN4 also includes `conf/training_asr_transformer.yaml`.
The TEMPLATE supplies missing keys for each selected role.

The baseline launcher retains only requested stage configs. Consequently,
standalone `--stages train` does not supply ASR's tokenizer config, and standalone
`--stages measure` does not supply the inference config needed for dataset
references. Keep `train_tokenizer` with `train`, and `infer` with `measure` as
shown. A system constructed directly from Python builds predecessor configs and
can execute only the desired method without re-running its predecessors.

Avoid `--stages all` for a training smoke test: it also selects model/demo
upload stages and requires their config roles and credentials.

For inference against an existing experiment:

```bash
python run.py --stages infer measure \
  --exp_dir exp/training \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Replace `exp/training` with the directory produced by your selected training
config. `--exp_dir` locates saved configs and supplies the experiment identity;
`--inference_dir` can explicitly choose an inference output directory. These
are supported directory flags. Arbitrary Hydra-style `key=value` overrides and
lists of files after one role flag are not supported.

ASR inference uses `ASRInferenceRunner` to unwrap contract-checked audio and
format `Speech2Text` n-best results. Declare `contract.inputs`/`outputs` in
`inference.yaml`; `input_key` must name exactly the declared inputs. The
contract path cannot also set the legacy `output_fn`.

Metrics bind their declared inputs separately, for example `hyp: text` reads
inference's `text.scp`, and `ref: dataset:text` reads the original test dataset.

Outputs live under the selected `exp_dir` and `inference_dir`. The TEMPLATE's
`stats_dir` is shared at `${recipe_dir}/exp/stats`; set an experiment-specific
path when model/data variants need independent statistics.

<details>
<summary>Programmatic execution and packing</summary>

Use `launch(..., argv=[...])` for programmatic CLI execution. For direct
single-method execution, load unresolved role configs and construct the system:

```python
from pathlib import Path
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.utils.config_utils import load_and_merge_config

roles = ("training", "inference", "metrics", "publication")
configs = {
    role: load_and_merge_config(
        Path("conf") / f"{role}.yaml",
        config_name=f"{role}.yaml",
        default_package="egs3.TEMPLATE.esp2_asr",
        resolve=False,
    )
    for role in roles
}
system = ASRSystem(configs=configs, stages_to_run=("pack_model",))
system.pack_model()
```

This builds predecessor configs without executing their stages; it requires a
trained model and the actual config paths for your recipe. The baseline CLI's
standalone `pack_model` request drops predecessor entries even if role flags are
passed, so use this direct construction when packing an existing run. Do not
call a removed `run.main(args=..., system_cls=...)` API.

`upload_model()`/`upload_demo()` publish to Hugging Face and require your chosen
repository and credentials. They are explicit publication operations, separate
from local packing.

</details>

## Next steps

<DocCards :cols="3">
  <DocCard
    title="What a Recipe Is"
    desc="Build a minimal recipe from scratch with your own data"
    icon="tabler:file-code"
    href="./what-is-a-recipe.html"
  />
  <DocCard
    title="Installation Troubleshooting"
    desc="CUDA mismatch, missing packages, and common errors"
    icon="tabler:tool"
    href="./installation-troubleshooting.html"
  />
  <DocCard
    title="System and Stages"
    desc="Learn how `run.py` and named stages fit together in ESPnet3."
    icon="tabler:puzzle"
    href="../stages/index.html"
  />
  <DocCard
    title="Config Overview"
    desc="See how training, inference, metrics, and publication configs connect."
    icon="tabler:settings-code"
    href="../config/index.html"
  />
</DocCards>
