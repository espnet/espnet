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

Earlier results from the removed ESPnet2 compatibility adapters describe the old
implementation only. They are not results for this stock ESPnet3 implementation.
New validation results are recorded separately; full accuracy reproduction is
not implied by a short pipeline test.

CPU validation for this revision: 20 recipe tests passed. Black, isort and flake8 passed.

A single-A10 run with the delivered 25-epoch maximum stopped after 23 epochs
through the configured early-stopping callback. It used CPU CTCLoss via an
experiment-only callback, fresh shared ESPnet3 statistics, and the existing
native AN4 LM at weight 0.1. All 4117 optimizer updates had finite gradients;
final parameters were finite. Inference used the stock top-1 average and scored
all validation/test recordings with the shared ESPnet3 metrics:

| Split | Recordings | WER (%) | CER (%) | TER (%) |
| --- | ---: | ---: | ---: | ---: |
| Validation | 100 | 15.57 | 10.29 | 9.78 |
| Test | 130 | 8.67 | 5.07 | 4.82 |

Training took about 16.8 minutes; scoring ran in a separate one-A10 allocation.
These results describe this stock ESPnet3 implementation, not the earlier
25-epoch ESPnet2-compatibility experiment or its SCTK scores.
