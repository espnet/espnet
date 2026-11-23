# VoxLingua107 LID recipe

Train an MMS/ECAPA-TDNN model to identify 107 spoken languages.

## Data Preparation

`create_dataset` downloads VoxLingua107 (the official zip files) and writes
the train and dev manifests to `data/voxlingua107`:

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

To reuse a corpus you already have, set `VOXLINGUA107` to it. Otherwise the
corpus is read from, or downloaded to, `download/voxlingua107` (relative to
this directory).

The additional evaluation sets (FLEURS, ML-SUPERB 2.0, VoxPopuli, Babel) are
prepared with the scripts described in [`prepare_corpora.md`](prepare_corpora.md).

## Commands

Run from `egs3/voxlingua107/esp2_lid` with ESPnet installed.

```bash
export VOXLINGUA107=/path/to/voxlingua107

# 1) Prepare the data, collect statistics, and train
python run.py --stages create_dataset collect_stats train \
    --training_config conf/training.yaml

# 2) Predict languages on the development set
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 3) Compute accuracy and macro precision, recall, and F1
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

To evaluate on a set prepared with `prepare_corpora.md`, set `prepared_name`
and `prepared_manifest` in `conf/inference_prepared.yaml` and run `infer` and
`measure` with that inference config.

Extract language embeddings and plot them with t-SNE:

```bash
python -m egs3.voxlingua107.esp2_lid.src.embeddings \
    --lid-train-config exp/training/config.yaml \
    --lid-model-file exp/training/valid.accuracy.best.pth \
    --lang2utt exp/stats/train/lang2utt \
    --manifest data/voxlingua107/dev/manifest.tsv \
    --output-dir exp/training/embeddings
```

## Results

### MMS-1B + ECAPA-TDNN

- Training configuration: [`conf/training.yaml`](conf/training.yaml)
  (MMS-1B multilayer features, ECAPA-TDNN encoder, AAM-softmax with
  sub-center and top-k, 30 epochs of 1,000 iterations, bf16 mixed precision).
  The checkpoint with the best VoxLingua107 dev accuracy is used.
- Inference configuration: [`conf/inference.yaml`](conf/inference.yaml),
  with FP32 matmul set to `highest`.
- Results: accuracy (%).

| Dataset | Split | Evaluated languages | Utterances | Accuracy (%) |
| --- | --- | ---: | ---: | ---: |
| VoxLingua107 | dev | 33 | 1,609 | 93.72 |
| FLEURS | test | 83 | 63,885 | 94.49 |
| ML-SUPERB 2.0 | dev | 91 | 17,631 | 88.31 |
| ML-SUPERB 2.0 | dev_dialect | 8 | 7,095 | 75.79 |
| VoxPopuli | test | 16 | 16,991 | 88.42 |

Evaluation includes all utterances whose reference language is supported by
the model; prediction candidates remain all 107 training languages.
VoxLingua107 dev is also used for checkpoint selection.

- Published model:
  [espnet/espnet3_lid_voxlingua107_mms_ecapa](https://huggingface.co/espnet/espnet3_lid_voxlingua107_mms_ecapa)
