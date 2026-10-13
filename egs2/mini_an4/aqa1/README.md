# Mini AN4 audio quality assessment

This CPU integration recipe trains Uni-VERSA for one epoch using the bundled
mini AN4 audio and **synthetic** UTMOS, PESQ, and WER labels. These scores are
fixtures, not measurements or a benchmark. Some labels are deliberately omitted
to exercise missing-label masking. Reference audio and character-tokenized text
are enabled.

From this directory, with ESPnet and sph2pipe installed:

```bash
./run.sh --ngpu 0 --stop_stage 9
```

Stages 1–9 prepare the data, collect statistics, train, infer, and score.
Predictions (`metric.scp`) and scoring results (`utt_result.json`) are written
under `exp/aqa_train_raw_fs16000/decode_valid.loss.best/<split>/`.
See [the AQA template](../../TEMPLATE/aqa1/README.md) for data and model options.
