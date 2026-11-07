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
- pretrained model: https://huggingface.co/espnet/frontia_vad_v1_online (upload pending)
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
- inference model: `valid.acc.best.pth` (epoch 11), whole recordings
- pretrained model (offline use): https://huggingface.co/espnet/frontia_vad_v1_offline (upload pending)
- tuned on ami_mh_dev in both modes: threshold 0.7, median filter 11

### Online (look-ahead 0, `conf/decode.yaml`)

| set | speech h | DetER (tuned) | FA | Miss | DetER (set's own best) | its threshold, median |
|---|---|---|---|---|---|---|
| aishell4_eval | 11.5 | 7.96 | 3.34 | 4.62 | 7.34 | 0.5, 11 |
| alimeeting_dev | 3.9 | 7.15 | 4.91 | 2.24 | 7.06 | 0.6, 11 |
| alimeeting_eval | 9.9 | 7.00 | 4.79 | 2.22 | 7.00 | 0.7, 11 |
| ami_dev | 75.1 | 8.23 | 2.70 | 5.53 | 7.55 | 0.5, 11 |
| ami_eval | 90.1 | 9.96 | 2.46 | 7.49 | 8.13 | 0.5, 11 |
| ami_mh_dev | 7.6 | 9.87 | 6.47 | 3.40 | 9.87 | 0.7, 11 |
| ami_mh_eval | 7.3 | 10.67 | 5.37 | 5.30 | 10.37 | 0.6, 11 |
| ava_speech_eval | 19.9 | 49.35 | 43.40 | 5.96 | 49.35 | 0.7, 11 |
| chime6_dev | 117.9 | 9.31 | 1.44 | 7.87 | 4.03 | 0.3, 11 |
| chime6_eval | 122.7 | 14.42 | 5.07 | 9.35 | 14.36 | 0.6, 11 |
| dev | 329.2 | 7.26 | 1.91 | 5.35 | 5.47 | 0.4, 11 |
| dipco_dev | 98.1 | 10.67 | 3.92 | 6.76 | 8.18 | 0.4, 11 |
| dipco_eval | 92.0 | 9.90 | 5.09 | 4.81 | 9.13 | 0.6, 11 |
| earnings21_eval | 32.8 | 17.29 | 11.02 | 6.27 | 17.29 | 0.7, 11 |
| icsi_dev | 12.5 | 11.78 | 5.18 | 6.60 | 11.78 | 0.7, 11 |
| icsi_eval | 11.9 | 14.72 | 3.72 | 11.00 | 13.49 | 0.5, 11 |
| notsofar_dev | 123.7 | 4.26 | 1.55 | 2.70 | 3.92 | 0.5, 11 |
| notsofar_eval | 398.5 | 4.03 | 2.44 | 1.59 | 3.91 | 0.6, 11 |
| ten_vad_eval | 0.1 | 18.60 | 16.90 | 1.70 | 18.60 | 0.7, 11 |
| voxconverse_dev | 18.9 | 4.60 | 3.36 | 1.25 | 4.60 | 0.7, 11 |
| voxconverse_eval | 39.0 | 8.04 | 6.86 | 1.17 | 8.04 | 0.7, 11 |

### Offline (look-ahead 124 frames = 0.99 s, `conf/tuning/decode_offline.yaml`)

| set | speech h | DetER (tuned) | FA | Miss | DetER (set's own best) | its threshold, median |
|---|---|---|---|---|---|---|
| aishell4_eval | 11.5 | 5.92 | 2.58 | 3.34 | 5.27 | 0.5, 11 |
| alimeeting_dev | 3.9 | 5.89 | 4.48 | 1.41 | 5.89 | 0.7, 11 |
| alimeeting_eval | 9.9 | 5.46 | 3.90 | 1.56 | 5.46 | 0.7, 11 |
| ami_dev | 75.1 | 5.96 | 2.10 | 3.86 | 5.65 | 0.6, 11 |
| ami_eval | 90.1 | 7.30 | 1.96 | 5.34 | 6.18 | 0.6, 11 |
| ami_mh_dev | 7.6 | 8.12 | 5.80 | 2.31 | 8.12 | 0.7, 11 |
| ami_mh_eval | 7.3 | 8.73 | 4.77 | 3.96 | 8.71 | 0.6, 11 |
| ava_speech_eval | 19.9 | 39.94 | 34.23 | 5.72 | 39.94 | 0.7, 11 |
| chime6_dev | 117.9 | 7.25 | 1.39 | 5.86 | 3.70 | 0.3, 11 |
| chime6_eval | 122.7 | 12.54 | 5.05 | 7.49 | 12.44 | 0.6, 11 |
| dev | 329.2 | 5.40 | 1.57 | 3.83 | 4.42 | 0.5, 11 |
| dipco_dev | 98.1 | 9.32 | 3.41 | 5.91 | 7.59 | 0.5, 11 |
| dipco_eval | 92.0 | 8.82 | 4.45 | 4.37 | 8.45 | 0.6, 11 |
| earnings21_eval | 32.8 | 10.89 | 8.83 | 2.07 | 10.89 | 0.7, 11 |
| icsi_dev | 12.5 | 10.42 | 4.48 | 5.94 | 10.42 | 0.7, 11 |
| icsi_eval | 11.9 | 13.79 | 3.56 | 10.23 | 12.74 | 0.5, 11 |
| notsofar_dev | 123.7 | 2.80 | 1.14 | 1.67 | 2.64 | 0.6, 11 |
| notsofar_eval | 398.5 | 2.82 | 1.79 | 1.03 | 2.82 | 0.7, 11 |
| ten_vad_eval | 0.1 | 18.10 | 16.90 | 1.21 | 17.98 | 0.7, 1 |
| voxconverse_dev | 18.9 | 4.23 | 3.05 | 1.18 | 4.23 | 0.7, 11 |
| voxconverse_eval | 39.0 | 7.51 | 6.57 | 0.94 | 7.51 | 0.7, 11 |
