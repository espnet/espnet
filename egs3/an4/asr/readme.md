# AN4 Sinc-BLSTMP ASR

Ports `egs2/an4/asr1/conf/train_asr_sinc_rnn.yaml`, AN4 data preparation and
the shared ASR shell stages. The original run.sh's default Transformer is not
the selected model.

## Run

Activate ESPnet and SoX, then run from this directory:

```bash
. ./path.sh
python run.py --stages create_dataset train_tokenizer collect_stats train \
  --training_config conf/training_sinc_rnn.yaml
python -m espnet3.systems.asr.language_model --config conf/language_model.yaml
python run.py --stages infer measure \
  --training_config conf/training_sinc_rnn.yaml \
  --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

The LM entrypoint reuses native ESPnet2 LMTask because ASRSystem has no LM
training stage. `conf/lm_native.yaml` is copied unchanged from the source.
It collects statistics, trains/averages checkpoints, and computes perplexity.
Inference uses the original LM weight 0.1 and evaluates validation plus test.

SCTK is required for the source-compatible WER/CER/TER. In an ESPnet checkout,
run `cd tools && bash installers/install_sctk.sh` once; the recipe's `path.sh`
adds the resulting `tools/sctk/bin` when present. Standalone recipes require
`sclite` on PATH.

## Source behavior and adapters

- First 100 sorted training IDs form validation; the remaining 848 receive
  0.9/1.0/1.1 speed perturbation. Original-speed IDs and original SoX options
  remain unchanged. Expected filtered ASR counts: 2544 / 100 / 130.
- Builder format 4 exports ASR manifests and original pre-filtering LM text.
  Rebuild older data and regenerate tokenizer/statistics in a new experiment.
- Unigram vocabulary 30; original Sinc-BLSTMP, Adadelta, plateau scheduler,
  25-epoch limit and best-validation-accuracy checkpoint selection.
- Folded batch size 15, fold lengths `[80000, 150]`, descending batch order,
  one data worker and native epoch seeds. Raw length scaling follows asr.sh.
- `lightning_module` selects the optional `ESPnet2LightningModule`.
  `espnet2_compat` owns accumulation/clipping, while Lightning trainer values
  remain 1/0 to avoid applying them twice. `distributed_batch_mode: split_batch`
  splits each global batch across ranks. cuDNN determinism and TF32 are explicit.
- `espnet2_compat.ctc_on_cpu: true` is a diagnostic override used by workspace
  GPU tests. Only CTCLoss runs on CPU; projection/log-softmax remain on GPU and
  cross-device autograd remains connected.
- Source early stopping `bad_epochs > 4` maps to Lightning patience 5.
- `espnet2_stats` preserves the source CPU collector, 32 contiguous splits,
  batch size 15, one worker, per-utterance accumulation and ordered merging.
- WER/CER/TER call the source SCTK scorer, including case handling, whitespace
  normalization and empty predictions.

## Validation

A paired ESPnet2/ESPnet3 run at base commit
`6f1263a6069de753cd0a888e341ff6e78e42ca96` completed 25 ASR epochs on one
NVIDIA A10 per run, with seed 0 and FP32 training. Both runs used the same
prepared audio, tokenizer, normalization statistics and native LM checkpoint
(LM weight 0.1). Only CTCLoss was moved to CPU in both runs to avoid CUDA CTC
nondeterminism; this is an experiment override, not the recipe default.

| Split | Utterances | WER (%) | CER (%) | TER (%) |
| --- | ---: | ---: | ---: | ---: |
| valid | 100 | 14.38 | 8.72 | 8.29 |
| test | 130 | 8.28 | 4.52 | 4.30 |

The final ASR checkpoints matched exactly across all 100 state tensors.
Decoding used the best validation-accuracy checkpoint. SCTK integer error
counts matched for all three metrics on both splits. Recognized text matched
after whitespace normalization; one test output had an extra trailing space.
This establishes single-GPU ASR parity under these conditions, not multi-GPU
parity or an independent comparison of two LM training runs.

Separate acceptance checks covered audio, tokenizer outputs, normalization
statistics and complete batch plans. AN4's original SoX dithering is stochastic:
exact waveform diagnostics fixed SoX randomness on both sides externally;
the recipe retains the source SoX command and its default randomness.
