# Mini AN4 LID recipe

A tiny ESPnet3 port of `egs2/mini_an4/lid1` that runs the whole LID pipeline
in a few minutes on CPU. It exists for integration tests, not for results.

AN4 has no language labels, so each speaker id is treated as a language, as
in the ESPnet2 recipe. The corpus is read from `../esp2_asr/downloads.tar.gz`,
and `sph2pipe` must be on `PATH`.

## Quick start

Run from `egs3/mini_an4/esp2_lid`.

```bash
source path.sh
python run.py --stages create_dataset collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
