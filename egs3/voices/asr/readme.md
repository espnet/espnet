# VOiCES Conformer ASR

Ports `egs2/voices/asr1/conf/train_asr_conformer.yaml` and its single-channel
source-plus-distant pipeline. `training_conformer.yaml` selects the full
VOiCES_rebuilt corpus. `training_devkit.yaml` selects the original devkit mode;
other parameters remain 5000 vocabulary, 35000000 batch_bins and 4 GPUs.
An existing extracted source can be selected with `create_dataset.source_dir`.
Otherwise the selected official archive is downloaded. The multi-channel data
variant is not implemented by this single-channel Dataset.

**The devkit text cannot train the source's 5000-piece unigram vocabulary:**
SentencePiece reports `Vocabulary size too high`. Workspace smoke tests explicitly
use a separate 1000-piece configuration. This is an experiment override, not a
change to the delivered source defaults. The full release has not been tested.

## Run with source settings

From this directory, after activating ESPnet:

```bash
. ./path.sh
python run.py --stages create_dataset train_tokenizer collect_stats train \
  --training_config conf/training_conformer.yaml
python -m espnet3.systems.asr.language_model --config conf/language_model.yaml
python run.py --stages infer measure \
  --training_config conf/training_conformer.yaml \
  --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

The explicit LM entrypoint reuses ESPnet2 LMTask because ASRSystem has no LM
training stage. `conf/lm_native.yaml` exactly copies the source Transformer LM.
The wrapper config includes the original external LibriSpeech text and runs
statistics, training, checkpoint averaging and perplexity. Inference loads the
LM at weight 0.6. A limited-text smoke test does not reproduce the full LM.

SCTK is required for the source-compatible WER/CER/TER. In an ESPnet checkout,
run `cd tools && bash installers/install_sctk.sh` once; the recipe's `path.sh`
adds the resulting `tools/sctk/bin` when present. Standalone recipes require
`sclite` on PATH.

## Source behavior and adapters

- First ten lexical training speakers form validation. Every recording variant
  stays with its source. ASR train/valid use strict 0.1--30 second filtering.
- Tokenizer/LM text is exported before duration filtering. Corrected devkit
  tokenizer text has 12540 lines, including the 33 previously missing lines.
  Builder format 2 invalidates old manifests. Audio is read in place.
- Original model, optimizer, warmup scheduler and AMP are retained.
  Descending batch order, one worker, native epoch seeding and disabled TF32
  are explicit. `ESPnet2LightningModule` owns accumulation 4 and clipping 5;
  Lightning's trainer accumulation/clipping remain 1/0. Incomplete gradients
  survive epoch boundaries as in the native trainer.
- `distributed_batch_mode: split_batch` splits each global sequence batch across
  ranks, retaining source batch membership and sample coverage.
- DDP uses the native `gradient_as_bucket_view: true`. The compatibility module
  restores DDP's buffer-synchronization flag before training forwards and the
  first validation forward, preserving BatchNorm behavior under Lightning
  manual optimization. DDP handles the actual broadcast.
- Metrics already reduced across ranks are accumulated in float64, matching
  the native reporter's Python-float weighted sums. This prevents float32
  rounding from changing checkpoint scores or scheduler inputs.
- `espnet2_compat.ctc_on_cpu: true` moves only CTCLoss to CPU. Workspace GPU
  tests enable it to isolate the previously observed CUDA CTC nondeterminism.
- Native AMP prefers BF16 on capable CUDA devices and otherwise FP16. The
  compatibility module reproduces this choice while retaining GradScaler;
  Lightning's `16-mixed` supplies the scaler, not the final model autocast dtype.
  Backward and optimizer updates run outside autocast, as in native Trainer.
- Inference sums the best 10 retained checkpoints in validation-accuracy order,
  using the final full `step*.ckpt` for exact scores. With fewer than 10, it
  selects the best one, as ESPnet2 does. Keep that full checkpoint alongside
  the retained weights. Earlier epochs take precedence when scores tie.
- `espnet2_stats` uses the source CPU collector, 32 contiguous splits, batch
  size 20, one worker and the original accumulation order. Independent devkit
  GlobalMVN counts, sums and squared sums match exactly in acceptance tests.
- WER/CER/TER use SCTK with the source tokenization, case handling, whitespace
  normalization and empty hypotheses.

For vocabulary overrides, update tokenizer.save_path, LM tokenizer_dir and TER
bpemodel together, and use a separate output directory. Previous small-batch,
1000-vocabulary, ASR-only scores are historical experiment results.

## Included compatibility code

This recipe includes its required adapters under `espnet3/`, alongside mirrored
unit tests. It runs from this branch without the AN4 recipe or AN4 checkpoints.
The base ESPnet3 behavior remains the default for recipes that do not opt in.
The measurement stage retains all WER/CER/TER outputs when a scorer class is
configured more than once.

## Validation scope

The full VOiCES release and a repaired 50-epoch accuracy reproduction have not
been validated. Experiments use the source devkit splits (12507 training,
660 validation and 6600 test recordings), with an explicit 1000-piece tokenizer
override and CPU CTCLoss. These overrides are separate from the delivered
full-corpus, 5000-piece defaults. The full stage integration test uses a small
CPU fixture; it does not establish recognition accuracy on the full release.

Standalone regression checks passed: 158 tests, with one CUDA-only test skipped
in the CPU test run. Formatting, import ordering and lint checks also passed.
On September 20, 2026, a paired single-node, four-A10 experiment completed one
devkit epoch in each framework: 117 training batches, 29 optimizer updates and
7 validation batches per rank. Inputs and random-number states matched across
all training batches, but gradients first differed at batch 66 and parameters
at batch 69. Final checkpoint tensors were not identical. Two native ESPnet2
runs also differed in gradients, starting at batch 94; this does not by itself
establish the cause of the migrated run's divergence. Strict numerical parity
remains unresolved, and this branch must not be described as a completed
accuracy reproduction.
