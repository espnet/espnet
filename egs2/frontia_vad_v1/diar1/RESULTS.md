# RESULTS

Detection error (%) = (missed + false alarm speech) / reference speech, md-eval,
collar 0 (with one pseudo speaker there is no speaker confusion, so the
diarization error rate is the detection error). Stage 7 scores every set at
thresholds 0.3-0.7 and median filters of 1 and 11 frames; `local/show_vad_result.py`
picks the setting on AMI Mix-Headset dev and reports every set at it ("tuned"),
next to each set's own best setting, which is tuned on that set.

## Causal model

- config: `conf/train.yaml`, 10 s windows, batch 64, learning rate 5e-4, seed 1
- inference model: `valid.acc.best.pth` (epoch 11), whole recordings, `conf/decode.yaml`
- pretrained model: TBD
- tuned on ami_mh_dev: threshold 0.7, median filter 11

| set | speech h | DetER (tuned) | FA | Miss | DetER (set's own best) | its threshold, median |
|---|---|---|---|---|---|---|
| aishell4_eval | 11.5 | 7.66 | 3.29 | 4.37 | 7.23 | 0.5, 11 |
| alimeeting_dev | 3.9 | 7.17 | 4.94 | 2.23 | 7.12 | 0.6, 11 |
| alimeeting_eval | 9.9 | 6.98 | 4.69 | 2.29 | 6.98 | 0.7, 11 |
| ami_dev | 75.1 | 8.04 | 2.72 | 5.33 | 7.47 | 0.5, 11 |
| ami_eval | 90.1 | 9.64 | 2.52 | 7.12 | 8.04 | 0.5, 11 |
| ami_mh_dev | 7.6 | 9.70 | 6.42 | 3.28 | 9.70 | 0.7, 11 |
| ami_mh_eval | 7.3 | 10.37 | 5.37 | 5.00 | 10.37 | 0.7, 11 |
| ava_speech_eval | 19.9 | 48.88 | 43.01 | 5.88 | 48.88 | 0.7, 11 |
| chime6_dev | 117.9 | 9.12 | 1.43 | 7.69 | 3.97 | 0.3, 11 |
| chime6_eval | 122.7 | 14.32 | 5.02 | 9.30 | 14.24 | 0.6, 11 |
| dev | 329.2 | 7.11 | 1.95 | 5.15 | 5.42 | 0.4, 11 |
| dipco_dev | 98.1 | 10.37 | 4.04 | 6.33 | 8.25 | 0.4, 11 |
| dipco_eval | 92.0 | 9.98 | 5.32 | 4.66 | 9.32 | 0.6, 11 |
| earnings21_eval | 32.8 | 14.55 | 10.33 | 4.23 | 14.55 | 0.7, 11 |
| icsi_dev | 12.5 | 11.70 | 5.26 | 6.44 | 11.70 | 0.7, 11 |
| icsi_eval | 11.9 | 14.50 | 3.80 | 10.70 | 13.43 | 0.5, 11 |
| notsofar_dev | 123.7 | 4.16 | 1.66 | 2.50 | 3.91 | 0.5, 11 |
| notsofar_eval | 398.5 | 3.96 | 2.41 | 1.55 | 3.86 | 0.6, 11 |
| ten_vad_eval | 0.1 | 17.44 | 16.10 | 1.34 | 17.44 | 0.7, 11 |
| voxconverse_dev | 18.9 | 4.53 | 3.48 | 1.05 | 4.53 | 0.7, 11 |
| voxconverse_eval | 39.0 | 7.91 | 6.98 | 0.93 | 7.91 | 0.7, 11 |

The tuned threshold is the top of the range stage 7 tries. The false alarm on
AVA-Speech (movies with music and effects) is far above every other set.

## Dual-mode model

- config: `conf/tuning/train_dualmode.yaml`, 10 s windows, batch 64, learning rate 5e-4, seed 1
- inference model: `valid.acc.best.pth`
- pretrained model: TBD

### Online (look-ahead 0, `conf/decode.yaml`)

Pending: the evaluation is running.

### Offline (look-ahead 124 frames = 1 s, `conf/tuning/decode_offline.yaml`)

Pending: the evaluation is running.
