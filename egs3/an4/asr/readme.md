# AN4 Sinc-BLSTMP ASR

Ports `egs2/an4/asr1` using the existing ESPnet3 `ASRSystem`, data loader,
statistics collector, Lightning module/trainer, checkpoint callbacks and metrics.
The recipe adds dataset preparation, configuration and inference output formatting;
it does not modify shared ESPnet3 code.

The first 100 lexical training utterance IDs form validation. Training audio uses
SoX speed perturbation at 0.9, 1.0 and 1.1. Train/valid use the source 0.1--20 second
filter; test recordings remain unfiltered. The full corpus contains 2544 prepared
training, 100 validation and 130 test recordings. SoX is required for preparation.

## Run

From this directory, activate an environment with ESPnet ASR dependencies:

```bash
. ./path.sh
python run.py --stages create_dataset train_tokenizer collect_stats \
  --training_config conf/training_sinc_rnn.yaml
python run.py --stages train --training_config conf/training_sinc_rnn.yaml
python -m egs3.an4.asr.src.language_model --config conf/language_model.yaml
python run.py --stages infer measure \
  --training_config conf/training_sinc_rnn.yaml \
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
the LM; inference uses that tokenizer and the trained LM at weight 0.1.

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

- `conf/training_sinc_rnn.yaml` retains the source Sinc frontend, BLSTMP/RNN,
  Adadelta, plateau scheduler, 30-piece unigram tokenizer and 25-epoch budget.
  Clipping is configured through Lightning. Its early-stopping patience is 5
  checks, corresponding to the source stopping after more than 4 bad epochs.
- The shared collector writes frontend `feats_shape`, not raw `speech_shape`
  and `text_shape`. Folded batches use an acoustic fold length of 334 frames
  (approximately 80000 input samples / 240-sample hop) and batch size 15.
  Text-length folding is no longer applied, so batch membership can change.
- Inference uses `Speech2Text` directly, the shared top-1 checkpoint average,
  beam size 10, CTC weight 0.3 and LM weight 0.1.

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

For review, compare data preparation with `egs2/an4/asr1/local/` and its
`run.sh`; compare ASR/LM model settings with `egs2/an4/asr1/conf/` and stage
defaults with `egs2/TEMPLATE/asr1/asr.sh`. ESPnet3 interfaces follow
`egs3/TEMPLATE/asr` and `egs3/librispeech_100/asr`.

## Validation

Recipe tests live under `test/egs3/an4/asr`, matching the source layout. Run
`pytest -q test/egs3/an4` from the repository root. They exercise preparation,
source model settings, output alignment, native LM commands and the stock ASR
stages on a small CPU fixture, including a trained LM with nonzero fusion weight.
Public recipe functions document their arguments, return values and usage examples.

Recipe validation: 20 tests passed; formatting, import order and lint checks passed.

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

The complete AN4 run used one NVIDIA A10, FP32, seed 0 and the configured
25-epoch maximum. Standard early stopping ended training after 21 epochs
(3759 optimizer updates); final parameters were finite. Inference used the
shared top-1 average, beam size 10 and LM weight 0.1 on one A10, covering all
100 validation and 130 test recordings. The following are the unchanged public
ESPnet3 WER/CER/TER outputs; TER uses the 30-piece ASR tokenizer.

| Split | Recordings | WER (%) | CER (%) | TER (%) |
| --- | ---: | ---: | ---: | ---: |
| Validation | 100 | 15.06 | 8.98 | 8.54 |
| Test | 130 | 8.93 | 5.03 | 4.86 |

### Local native ESPnet2 comparison

No matching published ESPnet2 result was found for this experiment. A local
baseline directly ran the official ESPnet2 preparation, statistics collection,
ASR training, inference and SCTK scoring at the same framework commit, using
GPU CTC and the same tokenizer and LM artifacts as the ESPnet3 run.
It used the original Sinc-RNN YAML on one A10 and completed 25 epochs with
the source early-stopping setting. The same 100 validation and 130 test IDs
and reference texts were verified. Native raw-speech/text folded batches and
statistics differ from ESPnet3's frontend-feature batches and shared statistics.

For a common scoring convention, the existing ESPnet3 hypotheses were separately
rescored with the same ESPnet2 tokenizer and SCTK used for the baseline. This
offline comparison does not change the recipe's public ESPnet3 metric classes
or the original predictions. Values below are percentages:

| Split | Framework | WER | CER | TER |
| --- | --- | ---: | ---: | ---: |
| Validation | ESPnet2 | 14.72 | 9.92 | 9.43 |
| Validation | ESPnet3, SCTK rescored | 15.06 | 8.98 | 8.54 |
| Test | ESPnet2 | 7.63 | 4.33 | 4.12 |
| Test | ESPnet3, SCTK rescored | 8.93 | 5.11 | 4.86 |

These are local native-framework results with different training trajectories
and stopping epochs, not a bitwise parity test. The scoring convention explains
why the rescored ESPnet3 CER differs slightly from its public-metric result.
