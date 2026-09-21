# VOiCES Conformer ASR

Ports the single-channel, source-plus-distant pipeline in `egs2/voices/asr1`
using the existing ESPnet3 `ASRSystem`, data loader, statistics collector,
Lightning module/trainer, checkpoint callbacks and metrics. Shared ESPnet3 code
is unchanged; this recipe does not depend on the AN4 recipe.

`training_conformer.yaml` selects the full VOiCES release. An extracted corpus
can be supplied through `create_dataset.source_dir`; otherwise the builder
fetches the configured official archive. The first ten lexical training speakers
form validation, with all recording conditions kept together. Train/valid use
the source 0.1--30 second duration filter; test audio remains unfiltered.
Tokenizer and LM text are exported before filtering. Cached manifests are
checked against their recorded hashes, and audio is read in place.

`training_devkit.yaml` selects the official devkit, with 12507 training, 660
validation and 6600 test recordings after filtering. It does not reduce the
vocabulary automatically: the devkit cannot train the source 5000-piece
vocabulary. For a devkit experiment, set `tokenizer.vocab_size: 1000` and
`tokenizer.save_path: ${data_dir}/unigram_1000` in a separate training config;
update the LM tokenizer path and the TER `bpemodel` as well. The full release and
multi-channel variant have not been validated.

## Run

From this directory, activate an environment with ESPnet ASR dependencies:

```bash
. ./path.sh
python run.py --stages create_dataset train_tokenizer collect_stats \
  --training_config conf/training_conformer.yaml
python run.py --stages train --training_config conf/training_conformer.yaml
python -m egs3.voices.asr.src.language_model --config conf/language_model.yaml
python run.py --stages infer measure \
  --training_config conf/training_conformer.yaml \
  --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

Run statistics collection and training as separate invocations, as above. At this
ESPnet3 revision the shared collector removes normalization from its in-memory
config; a fresh train invocation reloads GlobalMVN from YAML. Do not combine
`collect_stats train` in a single invocation.

ESPnet3 currently has no dedicated LM training stage. The recipe-local
`src/language_model.py` invokes existing ESPnet2 commands for LM statistics,
training and perplexity. `conf/lm_native.yaml` preserves the source LM model
settings. The helper records its commands in the LM experiment directory and
resumes an existing native checkpoint. Prepare the ASR tokenizer before training
the LM; inference uses that tokenizer and the trained LM at weight 0.6.

The VOiCES LM includes external LibriSpeech text. The source launcher uses four
GPUs, while its Transformer LM YAML describes a 500000000-bin budget used with
16 V100s. That full LM budget has not been validated here. Smaller experiments
must explicitly override it in their own config; testing a small LM does not
validate the full LM or its memory requirements.

## ESPnet2 mapping and ESPnet3 behavior

- The source Conformer/Transformer, SpecAugment, Adam settings, 40000 warmup
  steps, 5000-piece unigram tokenizer and 50-epoch budget are retained.
- The shared collector writes frontend `feats_shape`. The batch budget is
  2000000 feature elements per GPU, chosen for 24 GiB accelerators. The source
  35000000-bin budget counts raw speech and text across a global batch; that
  number cannot be reused as a per-GPU frontend-feature budget. A larger acoustic
  conversion (4375000) exceeded 24 GiB in devkit testing. Batch membership and
  updates per epoch differ from ESPnet2.
- The public loader assigns whole batches to ranks, truncating a tail of batches
  when necessary for equal rank lengths. `num_device: 4` and Lightning gradient
  accumulation 4 are explicit; changing GPU count changes the effective batch.
- Lightning owns clipping, accumulation boundaries and AMP (`bf16-mixed`).
  BF16 follows ESPnet2 on capable GPUs and the ESPnet3 LibriSpeech reference.
  This default requires BF16 support; older GPUs need a separate precision
  configuration. Remainder-gradient handling follows Lightning.
- Inference uses `Speech2Text` directly, the shared top-10 checkpoint average,
  beam size 20, CTC weight 0.3 and LM weight 0.6. Short experiments must explicitly
  select a checkpoint that exists; there is no recipe-specific averaging fallback.

The shared checkpoint callback writes `valid.acc.ave_Nbest.pth`, where `N` is
the number of retained checkpoints available to the averaging callback. For short
runs, inspect the experiment directory and explicitly set `model.asr_model_file`
in a separate inference config. `last.ckpt` is also produced by the stock trainer.
Averaging runs during validation and is not a separate finalization stage; the
average can precede the latest checkpoint update. This recipe uses that shared
callback without adding its own averaging logic. Keep checkpoints together with
their generated ASR `config.yaml`.

WER, CER and TER use the existing ESPnet3 metric classes (JiWER), with TER using
the ASR SentencePiece model. These are not the previous SCTK scorer: the shared
CER counts spaces and the shared metrics use a placeholder for empty text.
Trainer reductions, batching, precision, checkpoint averaging and metric
conventions can produce different results from ESPnet2. No strict numerical
identity with ESPnet2 is claimed.

For review, compare data preparation with `egs2/voices/asr1/local/` and its
`run.sh`; compare ASR/LM model settings with `egs2/voices/asr1/conf/` and stage
defaults with `egs2/TEMPLATE/asr1/asr.sh`. ESPnet3 interfaces follow
`egs3/TEMPLATE/asr` and `egs3/librispeech_100/asr`.

## Validation

Recipe tests live under `test/egs3/voices/asr`, matching the source layout. Run
`pytest -q test/egs3/voices` from the repository root. They exercise preparation,
source model settings, output alignment, native LM commands and the stock ASR
stages on a small CPU fixture, including a trained LM with nonzero fusion weight.
Public recipe functions document their arguments, return values and usage examples.

Earlier results from the removed ESPnet2 compatibility adapters describe the old
implementation only. They are not results for this stock ESPnet3 implementation.
New validation results are recorded separately; full accuracy reproduction is
not implied by a short pipeline test.

CPU validation for this revision: 764 tests passed across `test/espnet3` and
`test/egs3`; four were skipped (two optional Whisper imports and two CUDA-only
tests in the CPU test environment). Black, isort and flake8 passed.

GPU validation completed one devkit epoch on four NVIDIA A10 GPUs, using a
1000-piece tokenizer, BF16 mixed precision, 2000000 feature bins per GPU, and
CPU CTCLoss through an experiment-only callback. The shared loader/trainer ran
190 batches and 48 optimizer updates per rank; recorded gradients and final
parameters were finite. Peak allocated GPU memory was 13.22 GiB across ranks.
The run used `PYTORCH_ALLOC_CONF=expandable_segments:True`; this is an environment
setting, not a framework patch. The subsequent inference test loaded the
one-epoch weights directly, reused an existing native LM at weight 0.6, and
scored four validation plus four test recordings with the stock metrics.
This short run validates the pipeline, not recognition accuracy or the default
5000-piece/full-corpus/50-epoch setup. Tiny LM training and fusion were tested
separately in the CPU integration test.
