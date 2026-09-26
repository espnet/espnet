# Mini AN4 classification

A CI fixture, not a result-producing recipe. It runs every stage of
`espnet3.systems.cls.system.CLSSystem` — `create_dataset`,
`remove_long_short`, `prepare_labels`, `collect_stats`, `train`, `infer` and
`measure` — over seven utterances in a few seconds, so that a change to the
classification stages fails in `ci/test_integration_espnet3.sh` rather than in
a real recipe. See `egs3/meld/cls` for a recipe meant to produce results.

## Task

The class label is the AN4 speaker id, as in `egs2/mini_an4/lid1`, which uses
it as the language id. AN4 publishes no other per-utterance attribute.

| split | utterances | classes |
|---|---|---|
| train | 3 | `fash`, `fbbh`, `mwhw` |
| valid | 2 | `fash`, `mwhw` |
| test  | 2 | `fash`, `mwhw` |

AN4's test speakers (`fcaw`, `mmxg`) never appear in training, so the
classifier could not predict them. `lid1` relabels such utterances with a
randomly drawn training speaker; `dataset/config.yaml` uses a fixed table
instead, so the manifests are reproducible.

## Model

Log-mel features and a two-block transformer, trained for one epoch on one
batch. No upstream model is downloaded, so CI does not depend on the network.
The scores this produces are meaningless.

## Usage

```bash
source path.sh
python run.py \
    --stages create_dataset remove_long_short prepare_labels collect_stats \
             train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

The AN4 archive is read from `../asr/downloads.tar.gz`; nothing is downloaded.
