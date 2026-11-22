# mini_an4 esp2_asr recipe

A tiny end-to-end ASR fixture for exercising ESPnet3 stages and the ASR demo. Its one-epoch models are deliberately small and are not intended to produce meaningful recognition results.

## Data Preparation

The recipe uses the bundled `downloads.tar.gz` archive; `create_dataset` extracts it and builds train, validation, and test manifests. No separate download is needed.

## Commands

Run from this directory. This command prepares the data, trains a SentencePiece tokenizer and the small Transformer ASR model, then runs inference and WER/CER measurement:

```bash
source path.sh
python run.py \
  --stages create_dataset train_tokenizer collect_stats \
  --training_config conf/training_asr_transformer.yaml
python run.py --stages train infer measure \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Other configs exercise the streaming path (`conf/training_asr_streaming.yaml`) and the transducer paths (`conf/training_asr_transducer.yaml` and `conf/training_transducer_asr_conformer_rnnt.yaml`, the latter decoded with `conf/inference_transducer.yaml`). CI runs all of them and uses the streaming config to check that serial and two-worker inference give the same output.

Every config trains a tiny model for one batch of one epoch. Its recognition scores are not benchmark results, and no pretrained model is provided.
