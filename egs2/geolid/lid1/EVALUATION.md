# GeoLID evaluation details

See the [recipe README](README.md#results) for the results table and model links.

## Legacy and converted runtimes

A **converted model** is an original pretrained GeoLID checkpoint adapted
to the current ESPnet implementation by remapping parameter keys and removing
verified duplicate aliases. The learned tensor values are unchanged, and no
retraining was performed. These models use current ESPnet normalization and
pooling, so their predictions can differ from those of the original implementation.

The legacy-reference and converted columns report measured utterance-level
accuracy on all **1,609 VoxLingua107 dev utterances**, with correct counts in
parentheses for the converted models. The legacy reference evaluates the original
published weights through the earlier HF implementation, preserving legacy
normalization and pooling, with **batch size 4**. The converted models use the
current ESPnet implementation with **batch size 1**. Both evaluations use the same
utterance IDs and reference labels; the comparison includes both runtime and
batching differences. The other dataset columns in the paper rows, including
Macro Avg., are historical results. **Every combined paper number belongs to
the earlier epoch-24 checkpoint; the released combined checkpoint is epoch 31.**
The [results table](README.md#results) separates the historical combined row
from the released model.

## Evaluation settings and evidence

Macro Avg. is the unweighted mean of the six test sets' utterance-level
accuracies, rather than a mean over languages or accuracy pooled over all
utterances. New measurements use the full-precision accuracies before rounding;
historical rows retain the values reported in the paper. A dash denotes an
unreported measurement; no new six-set Macro Avg. is reported until all six
measurements exist.

The released combined row in the [results table](README.md#results) reports
new measurements at ESPnet revision
`07dc3256fbe6ab542a260314f9e3d9a0bdc9de9a`. The [combined evaluation
report](https://huggingface.co/espnet/geolid_combined_shared_trainable/blob/0461e9528b33bea08a29979cce60e685f04384c4/evaluations/20261027/report.md)
includes the reproduction command, source revisions, complete predictions,
reference labels and inference logs. The run covers all 1,609 VoxLingua107 dev,
77,810 FLEURS test, 23,609 ML-SUPERB 2.0 dev, 7,095 dialect-dev and 16,991
VoxPopuli test utterances. VoxPopuli uses the official test partition; the
recipe now reads `asr_test.tsv` instead of reusing development data.
ML-SUPERB uses the shared recipe's language normalization and excludes its
283 Norwegian dev rows. These split sizes, archived error IDs and reference
labels reproduce the historical reports' per-language accuracies. The old
reports do not establish byte-for-byte identity of all historical audio.

FLEURS uses the standard CLI's four-GPU multiprocessing mode, with one
utterance per forward pass and 8 workers per process. All 4,000 predictions
from the initial single-GPU run match the parallel run exactly. The other
test sets use one GPU.

**Babel remains unevaluated:** its licensed dev audio and original complete
manifest are unavailable in this evaluation environment. The released-model
Babel cell and six-set Macro Avg. remain blank pending that evaluation.

The new combined VoxLingua107 run reproduces all 1,609 previous predictions
(1,501 correct). The four VoxLingua107-only models' converted results retain
their original evaluation revision `301372e4da311d2a3d9574a5a43b3037ae6e67c1`.

These converted-model measurements use the standard `espnet2.bin.lid_inference`
CLI, GPU float32, **batch size 1 per forward pass**, 8 data-loading workers
per inference process and full-length waveforms
(`--use_preprocessor false --fix_duration false`). Inputs are sorted
by utterance ID, matching the recipe's `wav.scp`. The environment uses PyTorch
2.11.0+cu128, Transformers 5.14.1 and espnet-s3prl 2026.5.14. The recipe defaults
to `--inference_batch_size 1` for reproducible per-utterance evaluation.

Each HF repository preserves its earlier VoxLingua107 counts, runtime settings,
implementation revision and checkpoint hash in `evaluation.json`. The new
combined-model measurements and per-set evidence are under `evaluations/20261027/`.

## Combined checkpoint provenance

**All six combined-model paper accuracies and their Macro Avg. were obtained
with the earlier epoch-24 checkpoint (48,000 training batches), not the released
epoch-31 checkpoint (62,000 batches).** This applies to VoxLingua107 dev, Babel,
FLEURS, ML-SUPERB 2.0 dev, ML-SUPERB 2.0 dialect and VoxPopuli. The four
VoxLingua107-only models' legacy references remain consistent with their paper
VoxLingua107 dev results.

The [archived combined evaluation reports][combined-original-evaluations] give
94.41% on VoxLingua107 dev, 95.38% on Babel, 97.73% on FLEURS, 88.61% on
ML-SUPERB 2.0 dev, 86.82% on its dialect set and 98.95% on VoxPopuli. These round
to the combined paper row, whose reported Macro Avg. is 93.7%. All six inference
runs started between **00:34 and 02:37 CDT on June 2, 2025**; the last run, Babel,
finished at **02:55 CDT**. The logs record full-length audio, GPU float32,
batch size 4 and 32 workers, loading `valid.accuracy.best.pth`.

The [training log][combined-training-first] shows that the most recent best
checkpoint then was **epoch 24 / 48,000 training batches**. The
[resumed training log][combined-training-resumed] records a new best at
**epoch 31 / 62,000 batches**, at **11:22 CDT** that day, after the archived
evaluations. The published checkpoint's 41 BatchNorm `num_batches_tracked`
counters are all 62,000, consistent with that later checkpoint. Although
Section IV-C of the paper specifies a 62k checkpoint, every archived combined
score predates it. The epoch-24 weights were evaluated, while the later epoch-31
weights were released under the same filename.

This is a discrepancy between the historical results and the released checkpoint.
On VoxLingua107 dev, the gap from 94.4% is already present in the
**93.35% legacy reference**, before
conversion to the current **93.29%** model. The historical inference log did not
record a checkpoint hash, and the earlier checkpoint has not been recovered, so
its exact weight identity cannot be verified by checksum or a new run. The
epoch assignment above follows the archived training and inference logs.
The historical audio files were not checksummed either. All six combined paper
numbers and Macro Avg. remain historical epoch-24 results, not measurements of
the released epoch-31 weights.

[combined-original-evaluations]: https://huggingface.co/espnet/geolid_combined_shared_trainable/tree/7dc4bf988efff29f725bf34b1caa821c27359793/exp_combined/lid_mms_ecapa_upcon_32_44_it0.4_shared_trainable_raw/inference/valid.accuracy.best
[combined-training-first]: https://huggingface.co/espnet/geolid_combined_shared_trainable/blob/7dc4bf988efff29f725bf34b1caa821c27359793/exp_combined/lid_mms_ecapa_upcon_32_44_it0.4_shared_trainable_raw/train.4.log
[combined-training-resumed]: https://huggingface.co/espnet/geolid_combined_shared_trainable/blob/7dc4bf988efff29f725bf34b1caa821c27359793/exp_combined/lid_mms_ecapa_upcon_32_44_it0.4_shared_trainable_raw/train.3.log

The archived training logs retain epochs 24 and 31 while removing other epoch
files. This supports checking the original experiment directory for
`24epoch.pth`; it does not establish whether that file is still available today:

```text
exp_all_no_filter_raw/spk_mms_ecapa_upcon_32_44_it0.4_sharedCondProj_butUpdate_50k_lr1e-5_datasetup0.3_raw/24epoch.pth
```

A controlled epoch-24/epoch-31 comparison with the current recipe remains
outstanding. It requires recovering the epoch-24 weights and evaluating both
checkpoints on identical inputs with batch size 1. The historical epoch-24 row
is not such a rerun; recovering the Babel audio and complete manifest is also
necessary to report all six datasets and their Macro Avg.

## Batch dependence

`utterance_mvn` and final `chn_attn_stat` pooling both mask padding. The remaining
batch dependence arises **before final pooling**:

- S3PRL reports `ceil(samples / 320)` frames and extends the full batch's hidden
  states by repeating its last frame. For a shorter item, frames beyond its
  valid Wav2Vec2 convolution output can instead be padded-query outputs from
  the longer batch. The utterance-level length mask includes these extra frames.
- ECAPA-TDNN does not take frame lengths: its temporal convolutions and SE
  global averages operate over the padded time axis. Adding zero padding to
  identical valid input features changes the valid encoded features and the
  embedding, even when final pooling receives the correct length mask.

These effects were isolated using the same speech in single-utterance and
mixed-length batches, and by padding fixed frontend features before ECAPA.
Masking only the final pooling cannot undo those upstream feature changes.

## Converting an original checkpoint

```bash
python local/convert_checkpoint.py \
    --checkpoint /path/to/original/valid.accuracy.best.pth \
    --config /path/to/original/config.yaml \
    --languages /path/to/languages.txt \
    --output-dir /path/to/converted
```

Use an empty output directory and one ISO-639-3 code per line in classifier-index
order. The script supports the original shared, independent and early frozen
projection names, and the earlier converted layout. It moves the auxiliary head
from `loss.lang2vec_head.0.*` to `lang2vec_head.*` and moves auxiliary options
from `loss_conf` to `model_conf`. It also updates earlier `model: lang2vec`
configurations to `model: espnet`, preserving their auxiliary loss settings
and already converted parameter names. Legacy key handling exists only in
this conversion script.
