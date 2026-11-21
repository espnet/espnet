# mini_an4 esp2_cls recipe

A CI fixture for the ESPnet3 classification stages, not a result-producing recipe. It uses seven AN4 utterances and a tiny model for a one-batch, one-epoch run. For a classification recipe intended to produce results, see [MELD](../../meld/esp2_cls/README.md).

## Data Preparation

The recipe reads the bundled AN4 archive at `../esp2_asr/downloads.tar.gz`. `create_dataset` extracts the source and builds the manifests. The class label is the speaker ID; the two test speakers are mapped to training labels because they are unseen during training. The reproducible split is:

| Split | Utterances | Labels |
| --- | ---: | --- |
| Train | 3 | `fash`, `fbbh`, `mwhw` |
| Valid | 2 | `fash`, `mwhw` |
| Test | 2 | `fash`, `mwhw` |

## Commands

Run from this directory. The command exercises dataset preparation, duration filtering, label preparation, statistics collection, training, inference, and measurement:

```bash
source path.sh
python run.py \
  --stages create_dataset remove_long_short prepare_labels collect_stats \
  --training_config conf/training.yaml
python run.py --stages train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

This one-epoch, one-batch model checks that the classification pipeline runs. The resulting scores are not meaningful, and no pretrained model is provided.
