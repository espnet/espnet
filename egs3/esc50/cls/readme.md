# ESC-50 environmental sound classification recipe

Classifies each 5-second clip into one of 50 everyday sound categories (`dog`,
`rain`, `chainsaw`, ...) with a [BEATs](https://arxiv.org/abs/2212.09058)
encoder and a linear head. Port of `egs2/esc50/asr1`, which predates ESPnet's
classification task and had to express the problem through `asr.sh`.

[ESC-50](https://github.com/karolpiczak/ESC-50) ships 2,000 clips at 44.1 kHz
while BEATs needs 16 kHz, so `create_dataset` resamples each clip once into
`ESC50_OUTPUT` and writes one manifest per split. `ffmpeg` must be on `PATH`.
The corpus itself is only ever read, so a read-only mount is fine;
`create_dataset` reports where to put it rather than downloading it.

## Requirements

- ESC-50, holding `meta/esc50.csv` and `audio/`.
- The `BEATs_iter3` checkpoint from
  <https://github.com/microsoft/unilm/tree/master/beats>. No automatic download.
- `pip install -e ".[cls]"` and `ffmpeg`.

```bash
export ESC50=/path/to/ESC-50-master      # corpus (read-only is fine)
export ESC50_OUTPUT=/path/to/scratch     # resampled audio, manifests, token_list
export BEATS_CKPT=/path/to/BEATs_iter3.pt
```

`ESC50_OUTPUT` is optional and defaults to `data/`; the audio needs about
320 MB. `BEATS_CKPT` is written into `${exp_dir}/config.yaml` at training time
and read again by `infer`, so it must resolve wherever inference runs.

## Splits

ESC-50 ships five cross-validation folds. This recipe reproduces
`egs2/esc50/asr1/local/data_prep_multi_folds.py`: one fold is held out and the
other four train. `ESC50_FOLD` selects it and defaults to 1.

| Split | Clips | Source |
|---|---|---|
| train | 1600 | the four folds that are not `ESC50_FOLD` |
| valid | 400 | fold `ESC50_FOLD` |
| test | 400 | fold `ESC50_FOLD` |

`valid` and `test` are the same clips, as in ESPnet2, where `run.sh` passes the
held-out fold as both `valid_set` and `test_set`. Model selection therefore
happens on the fold being scored, which flatters the result; it is kept so the
numbers stay comparable to the ESPnet2 README.

All 50 categories appear in every split, and each fold holds exactly 8 of each.
The resampled audio does not depend on the fold, so the five folds share one
`${ESC50_OUTPUT}/wav` directory and the clips are converted once. Manifests,
`token_list`, and `exp/` are per fold.

## Quick start

Every stage reads `ESC50_FOLD`, so one fold is one pass through the pipeline:

```bash
export ESC50_FOLD=1

# 1) Resample, write manifests, filter, build the label list
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats --training_config conf/training.yaml

# 3) Train
python run.py --stages train --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

For the five-fold average, run that once per fold and mean the test WA. The
folds are independent, so give each its own GPU if you have five.

```bash
for fold in 1 2 3 4 5; do
    ESC50_FOLD=$fold python run.py \
        --stages create_dataset remove_long_short prepare_labels \
                 collect_stats train infer measure \
        --training_config conf/training.yaml \
        --inference_config conf/inference.yaml \
        --metrics_config conf/metrics.yaml
done

python - <<'EOF'
import json, pathlib
wa = [
    json.loads(p.read_text())["espnet3.systems.cls.metrics.wa.WA"]["test"]["WA"]
    for p in sorted(
        pathlib.Path("exp").glob("train_cls_beats_iter3_fold*/inference/metrics.json")
    )
]
print([f"{v:.2f}" for v in wa], f"avg {sum(wa) / len(wa):.2f}")
EOF
```

Every clip is exactly 5.00 s, so `remove_long_short` drops 0 of 2000. It is kept
because ESPnet2 runs it and a non-zero drop means a broken corpus. It is not
optional: `prepare_labels` and training both read `manifest_filtered/`.

Training one fold needs one GPU with about 40 GB; a 40 GB A100 peaks at 34 GB
and takes roughly 2.5 h. On a smaller card, halve
`dataloader.train.iter_factory.batches.batch_size` and set
`trainer.accumulate_grad_batches: 2`. The five folds are independent, so they
run side by side given five GPUs.

`max_epochs` is 700 rather than ESPnet2's 1000: 1600 clips at batch 128 is 13
steps/epoch, so the 6000-step cosine cycle bottoms out near epoch 460. The cap
is not binding -- across the five folds the best epoch was 209, 541, 554, 615,
and 646.

## Results

`measure` writes all five metrics to `${inference_dir}/metrics.json` per fold.

| Fold | WA | UA | Macro F1 | mAP | AUC | ESPnet2 WA |
|---|---|---|---|---|---|---|
| 1 | 94.00 | 94.00 | 93.57 | 95.77 | 99.61 | 94.3 |
| 2 | 95.75 | 95.75 | 95.68 | 98.41 | 99.96 | 97.0 |
| 3 | 93.75 | 93.75 | 93.53 | 97.00 | 99.87 | 94.8 |
| 4 | 95.75 | 95.75 | 95.68 | 98.47 | 99.94 | 96.3 |
| 5 | 91.50 | 91.50 | 91.18 | 95.33 | 99.54 | 91.8 |
| **avg** | **94.15** | 94.15 | 93.93 | 97.00 | 99.78 | **94.8** |

WA and UA are equal everywhere because each fold holds exactly 8 clips of each
of the 50 classes, so weighting by class frequency changes nothing.

The 94.15 average is 0.65 below ESPnet2's 94.8, and the per-fold spread tracks
it closely: fold 5 is the hardest in both (91.50 vs 91.8) and folds 2 and 4 the
easiest. The gap is not the shorter epoch budget -- every fold peaked between
epochs 209 and 646, inside the 700 cap -- so what is left is run-to-run variance
(std 1.76 across folds) and the trainer differences between ESPnet2 and
Lightning.

Each fold's peak `valid/acc` equals its test WA exactly, because `valid` and
`test` are the same clips.
