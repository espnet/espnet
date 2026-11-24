# Mini AN4 LID recipe

A tiny ESPnet3 port of `egs2/mini_an4/lid1` that runs the whole LID pipeline
in a few minutes on CPU. It exists for integration tests, not for results.

## Data Preparation

`create_dataset` reads the archive of the ASR recipe in place
(`../esp2_asr/downloads.tar.gz`), so no separate download is needed.
`sph2pipe` must be on `PATH`.

AN4 has no language labels, so each speaker id is treated as a language, as
in the ESPnet2 recipe. Test speakers never appear in training and are mapped
to fixed training speakers (`test_lang_map` in `dataset/config.yaml`).

## Commands

Run from `egs3/mini_an4/esp2_lid`.

```bash
source path.sh
python run.py --stages create_dataset collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Results

### ECAPA-TDNN (tiny)

- Training configuration: [`conf/training.yaml`](conf/training.yaml), a tiny
  ECAPA-TDNN trained for one epoch of 4 iterations.
- Results: not meaningful; the scores only show that every stage ran.
- Published model: none.
