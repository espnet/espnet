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

CTC loss follows the model/input device, as in ESPnet2. GPU training uses the
existing GPU CTC implementation. GPU CTC can introduce nondeterministic
differences and does not promise bitwise agreement across training runs.

The recipe uses the shared ESPnet3 training flow, including Lightning's default
pre-training validation checks. It does not replay initial weights, reset RNG
state at epoch boundaries, force ESPnet2 batch membership, replace gradient
accumulation/clipping, or override the shared precision and statistics code.
The configured seed and source model/data settings are ordinary recipe inputs.
No experimental loss or gradient-monitoring callback is required to run it.

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
  BF16 follows the ESPnet3 LibriSpeech reference and requires BF16 support;
  older GPUs need a separate precision configuration. The local ESPnet2
  baseline below uses the source FP16 AMP setting. Remainder-gradient handling
  follows Lightning.
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

Recipe validation: 36 tests passed; formatting, import order and lint checks passed.

### Native GPU CTC results

Validation used the shared framework at commit
`6f1263a6069de753cd0a888e341ff6e78e42ca96`, Python 3.13.2,
PyTorch 2.11.0+cu128, Lightning 2.6.0 and JiWER 4.0.0. The runs below use
native GPU CTC and Lightning's default pre-training validation, with fresh
ASR initialization and no numerical-alignment or diagnostic callbacks.
Previously prepared data, shared ESPnet3 statistics, tokenizer and native
ESPnet2 LM artifacts were reused; the LM was not retrained in these runs.
Earlier CPU CTC and compatibility-adapter experiments are historical diagnostics
and are not included in the results below.

The devkit run used four NVIDIA A10 GPUs for 50 epochs, BF16 mixed precision,
seed 0, 2000000 feature elements per GPU and gradient accumulation 4. Each rank
completed 2400 optimizer updates; final parameters were finite. Experiment
configs selected the devkit corpus, a 1000-piece unigram tokenizer and
matching TER/LM tokenizer paths, with the existing devkit-only LM and no external
LibriSpeech text. The standard 40000-step warmup was retained. The full-release
configuration and its 5000-piece vocabulary are unchanged.

Inference used one A10, batch size 4, the shared top-10 average, beam size 20 and
LM weight 0.6. Both the 660-recording validation set and 6600-recording test set
were explicitly selected in the experiment config, with all IDs and reference
texts verified. `PYTORCH_ALLOC_CONF=expandable_segments:True` was used for GPU
memory allocation. The following are the unchanged public ESPnet3 metrics:

| Split | Recordings | WER (%) | CER (%) | TER (%) |
| --- | ---: | ---: | ---: | ---: |
| Validation | 660 | 92.84 | 70.90 | 91.67 |
| Test | 6600 | 93.40 | 71.79 | 91.79 |

### Local native ESPnet2 comparison

No matching published ESPnet2 result was found for this experiment. A local
baseline directly ran the official ESPnet2 preparation, statistics collection,
ASR training, inference and SCTK scoring at the same framework commit, using
GPU CTC and the same tokenizer and LM artifacts as the ESPnet3 run.
It completed 50 epochs on two A10 GPUs with native FP16 AMP, raw-speech/text
numel batches and gradient accumulation 4. The source 35000000-bin budget
exceeded two 24 GB GPUs, so the experiment explicitly used
`--asr_args "--batch_bins 17500000"`; source model YAML and trainer code were
unchanged. This batch adjustment, GPU count, precision, batch distribution and
optimizer update counts differ from the ESPnet3 run.

The official final 10-best average ran separately in a fresh CPU process after
host-memory exhaustion in the training process. It used the same saved reporter
and `valid.acc/max` selection; the training checkpoint hash was unchanged and
averaged parameters were finite. Full inference and scoring then completed
successfully on one A10. This recovery did not change CTC or repeat training.

For a common scoring convention, the existing ESPnet3 hypotheses were separately
rescored with the same ESPnet2 tokenizer and SCTK used for the baseline. This
offline comparison does not change the recipe's public ESPnet3 metric classes
or the original predictions. Values below are percentages:

| Split | Framework | WER | CER | TER |
| --- | --- | ---: | ---: | ---: |
| Validation | ESPnet2 | 91.01 | 69.29 | 91.33 |
| Validation | ESPnet3, SCTK rescored | 92.90 | 71.16 | 91.83 |
| Test | ESPnet2 | 92.35 | 70.37 | 91.54 |
| Test | ESPnet3, SCTK rescored | 93.51 | 72.01 | 91.90 |

The high devkit error rates also occur in the native ESPnet2 baseline. This
validates full-budget devkit training, inference and scoring as a migration
check, not full-corpus recognition quality. Similar high errors alone do not
prove every migration detail correct; the framework and experiment differences
above remain part of the comparison.
