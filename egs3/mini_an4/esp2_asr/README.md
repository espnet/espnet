# mini_an4 esp2_asr recipe

A tiny end-to-end ASR fixture for exercising ESPnet3 stages and the ASR demo. Its one-epoch models are deliberately small and are not intended to produce meaningful recognition results.

## Data Preparation

The recipe uses the bundled `downloads.tar.gz` archive; `create_dataset` extracts it and builds train, validation, and test manifests. No separate download is needed.

## Commands

Run from this directory. This command prepares the data, trains a SentencePiece tokenizer and the small Transformer ASR model, then runs inference and WER/CER measurement:

```bash
source path.sh
python run.py \
  --stages create_dataset train_tokenizer collect_stats train infer measure \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Other training configs exercise streaming ASR and transducer variants: `training_asr_streaming.yaml`, `training_asr_transducer.yaml`, and `training_transducer_asr_conformer_rnnt.yaml`. The latter uses `conf/inference_transducer.yaml`. CI also checks serial and two-worker inference equivalence.

## Results

### Tiny Transformer ASR

`conf/training_asr_transformer.yaml` runs a tiny model for one batch and one epoch. Its recognition scores are not benchmark results; no pretrained model is provided.

### Streaming ASR

`conf/training_asr_streaming.yaml` exercises the streaming path. CI additionally compares serial and two-worker inference outputs.

### Transducer ASR

`conf/training_asr_transducer.yaml` and `conf/training_transducer_asr_conformer_rnnt.yaml` exercise transducer paths; the latter uses `conf/inference_transducer.yaml`. These are CI fixtures, not benchmark configurations.
