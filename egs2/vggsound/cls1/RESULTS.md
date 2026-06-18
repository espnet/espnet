# RESULTS

## Task

This recipe performs single-label audio classification on VGGSound with 309
classes. Training uses multi-class classification. Evaluation reports top-1
accuracy and mean one-vs-rest AP/AUC over classes from posterior scores.

## Environments

- date: `Thu Jun 18 08:11:42 UTC 2026`
- python version: `3.10.14 (tags/v3.10.14-25-ge98930d7387-dirty:e98930d7387, May 24 2024, 23:30:09) [GCC 13.2.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu128`
- Git hash: `dfe5cfca2ac5a8e50aeb2d1cb9dd97e56e057032`
  - Commit date: `Thu Jun 4 19:52:29 2026 -0400`

## Recipe

- config: `conf/beats_cls_vggsound.yaml`
- command:

```sh
./run.sh --stage 1 --stop_stage 8 --ngpu 1
```

## Results

### cls_20260611.101859

|Split|mean_acc|mAP|mean_auc|n_labels|n_instances|
|---|---|---|---|---|---|
|cls_test|56.07|58.59|96.62|309.00|14790.00|
|cls_valid|77.72|77.16|99.34|309.00|17568.00|
