# Audio quality assessment: Uni-VERSA and ARECHO checkpoint compatibility

The `aqa` (audio quality assessment) task revives the regression models from
[PR #5959](https://github.com/espnet/espnet/pull/5959) and the autoregressive
ARECHO model from `ftshijt/espnet`'s `universa_plus` branch. The task name describes
the function; model names and parameter prefixes retain `universa` for checkpoint
compatibility. See the [recipe guide](../egs2/TEMPLATE/aqa1/README.md)
for data preparation and training commands.

## Review findings and fixes

- Both historical models returned a negated training loss. Minimizing that value
  maximized prediction error. Training now returns the positive objective.
- Regression label indexing and single-branch projector construction were broken.
  Tests cover both branch layouts, pooling choices, missing labels, and backward.
- Pooling now uses ESPnet's existing `feat_lengths` interface. Padding must not
  affect predictions; no speaker-pooling implementation changes are needed.
- Disabling normalization left features uninitialized. Missing whole reference
  modalities now retain zero-filled embedding slots so projector shapes match.
  Reference-text padding is masked before embedding without mutating inputs.
- Metric discovery/evaluation now uses the union of utterance metric names.
  Missing labels are masked, including batches with no labels for a metric.
  A missing reference waveform is handled before attempting audio-file reads.
- ARECHO search restricts top-k to valid metric tokens, handles beams larger than
  the candidate set, preserves accumulated scores when skipping meta-label scores,
  and supports fixed metric order. Categorical token offsets are consistent on
  encoding and decoding, including the released `language` vocabulary.
- Configs embed metric names, type metadata, and tokenizer metadata when saved.
  Legacy file paths remain accepted. Inference emits numeric or categorical JSON
  values; requested token sequences are written separately to `token.scp`.
- Recipes use shared symlinks, `train.yaml` / `decode.yaml`, and `conf/tuning` for
  alternatives. Historical ablation configs remain in the original branch.

## Compatibility contract

Do not rename `universa.*` state-dict keys or reinterpret the historical
`embedding_dim` and `use_rope` config fields. The old constructors ignored those
fields: released weights use the original embedding shapes and sinusoidal
positions. New configs use `embedding_size`; actual RoPE is not implemented.
ARECHO preserves the trained SOS=2 / EOS=3 convention even though the historical
textual tokenizer labels name those IDs in the opposite order.

New commands use `espnet2.bin.aqa_train`, `espnet2.bin.aqa_inference`,
`espnet2.tasks.aqa.AqaTask`, and the `egs2/*/aqa1/aqa.sh` recipe driver.
The unused legacy training/task/model-package, packing, and evaluator aliases
have been removed, together with their compatibility-only tests. This PR adds
no legacy recipe or shell-script links.

### Consumer migration status

The public entry-point migration is complete: the legacy inference shim has
also been removed following
[the migration agreement in #6827](https://github.com/espnet/espnet/pull/6827#issuecomment-6047947493).
Use `espnet2.bin.aqa_inference.AqaInference` for Uni-VERSA and ARECHO.
The affected consumers and their migration status are:

| Consumer | Completed migration |
| --- | --- |
| VERSA Uni-VERSA and ARECHO loaders | [VERSA PR #106](https://github.com/wavlab-speech/versa/pull/106) is merged in [8b931fb](https://github.com/wavlab-speech/versa/commit/8b931fb2350c5a17a9d5b114817bd4262fbad47d). Both imports and the ESPnet installation pin use AQA; inference diagnostics are excluded from scores. |
| Six ESPnet Uni-VERSA model cards | Published canonical AQA examples on 2026-10-15, replacing the historical Colab link with the AQA guide. See the [no-reference card commit](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_noref/commit/ed903eaa4f2e3552056e4cc12a8aaf6e0e3b5d10) and [older-package artifact notice](https://huggingface.co/espnet/universa-base_urgent24_multi-metric/commit/d16bb2a5bb222fd7ecc710c03a7af34bc8e4ae8b). |
| Four ESPnet ARECHO model cards | Published the AQA API and `aqa1` / `aqa.sh` instructions on 2026-10-15, replacing the historical recipe/checkout instructions. See the [base card commit](https://huggingface.co/espnet/arecho_base_v0/commit/f3ec00dce7b1152f2490a1cff040db94bfe51ec5) and [scale card commit](https://huggingface.co/espnet/arecho_scale_v0/commit/69dee7470c6cc9e1990638482dfc152acc27f364). |

The six Uni-VERSA cards are `espnet/universa-base_urgent24_multi-metric`,
`espnet/universa-wavlm_base_urgent24_multi-metric`, and the latter's `_noref`,
`_audioref`, `_textref`, and `_fullref` releases. The four ARECHO cards are
`espnet/arecho_base_v0`, `espnet/arecho_scale_v0`,
`espnet/arecho_base_v0.1-large-decoder`, and
`espnet/arecho_scale_v0.1-large-decoder`.

The merged VERSA consumers and published instructions no longer depend on legacy
ESPnet entry points. Canonical AQA tests cover checkpoint loading, training, and
inference, and consumer smoke tests exercise all four Uni-VERSA reference modes
and ARECHO with tiny checkpoints. These checks do not establish that every
published large-weight variant has completed strict loading, save/reload,
fine-tuning, and inference; the published-weight validation limits below remain
separate from the completed public naming migration.
All ten live model cards were read back and matched the prepared updates; their
original configuration sections are unchanged. The cards now point to the
canonical inference example below instead of the historical Colab. The old
notebook itself remains historical: it installs a separate fork and does not
consume the removed task/model-package wrappers on current ESPnet. Historical artifact filenames such as `universa_train_.../13epoch.pth`
are actual published paths, not public recipe entry points; do not rename them.

### Canonical inference example

Install current ESPnet with the AQA task on Python 3.12 or 3.13 and install
`espnet-s3prl` for the released WavLM frontend. For the complete no-reference
release:

```python
import librosa
import soundfile as sf

from espnet2.bin.aqa_inference import AqaInference

audio, sample_rate = sf.read("example.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
predict = AqaInference.from_pretrained(
    "espnet/universa-wavlm_base_urgent24_multi-metric_noref"
)
scores = predict(audio)
print({name: scores[name] for name in predict.universa.metric2id})
```

For the complete `_audioref`, `_textref`, and `_fullref` releases, select the
corresponding model ID and pass `ref_audio` (a mono waveform at 16 kHz),
`ref_text` (the transcript), or both. ARECHO uses the same class and its own
model ID, with one utterance per batch. `encoded_feat`, `token_seq`,
`use_tokenizer_metrics`, and `sequential_metrics` are inference diagnostics,
not predicted metric scores. Package-relative artifact paths must be resolved
before loading a manually downloaded config.

Use the original architecture config with resolved artifact paths and
`--init_param /path/to/model.pth` to initialize fine-tuning. Preserve `freeze_param`
to freeze the same modules as the released config. An epoch weight file alone
cannot restore optimizer/scheduler state; `--resume true` needs a full trainer
checkpoint. New training intentionally uses the corrected positive objective.

## Published checkpoint validation

During the original series validation, the following representative weights were
strictly loaded (no missing/unexpected keys), then trained for one optimizer step
with their configured frozen parameter
prefixes. The check requires finite loss and gradients, verifies a trainable
parameter changed, and requires one valid prediction for every requested metric.

| Model | Hugging Face revision | Weight |
| --- | --- | --- |
| [Uni-VERSA WavLM no-reference](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_noref) | `6abec38c0971c2c0feebbe39ee2f94c9c21357c4` | `update_exp/universa_train_universa_wavlm_noref_raw_fs16000/13epoch.pth` |
| [ARECHO base v0](https://huggingface.co/espnet/arecho_base_v0) | `e6619474fc7b462db8da13183c9b565778c07980` | `exp/universa_universa_ar_overall_base_token_wavlm/68epoch.pth` |

Run the opt-in check from the repository root after downloading the model package
at the recorded revision and preparing the WavLM cache:

```bash
PYTHONPATH=. python test/espnet2/aqa/check_published_checkpoint.py \
  /path/to/arecho_base_v0 \
  --config exp/universa_universa_ar_overall_base_token_wavlm/config.yaml \
  --checkpoint exp/universa_universa_ar_overall_base_token_wavlm/68epoch.pth \
  --frontend-cache /path/to/wavlm-cache \
  --metrics '{"srmr": 2.5, "language": "<eng>"}'
```

For the Uni-VERSA package, use its paths from the table and `--metrics '{"mos": 2.5}'`.
The check reads local artifacts; it does not download model packages. The frontend
library may fetch its upstream model if the supplied cache is incomplete.
Relative artifact paths, including a file-backed `token_list`, are resolved from
the package root; embedded metadata is used directly. `--metrics` must be a
non-empty JSON object with known metric names and finite, unmasked numeric labels
or categorical labels from the released vocabulary. Models configured for
reference audio or text receive synthetic references, including an unknown text
token. This checks execution and parameter updates, not reference quality.

The original published-weight validation used CPU PyTorch 2.11 and Python 3.10, with
current `espnet-s3prl`. This is a compatibility smoke check, not a quality benchmark
or validation of every checkpoint. Current ESPnet's supported Python 3.12/3.13
matrix and GPU runs remain CI work. A separate tiny model completed ESPnet
statistics collection and one epoch of training. Full shell recipes were syntax
checked but not executed during that validation because sph2pipe was unavailable.
These historical results do not replace a rerun against the current revision.
Small local regression tests exercise the validator with both architectures and
with and without references, including strict-load and inference failure cases.

Artifact gaps found in the supplied collections:

- `espnet/universa-base_urgent24_multi-metric` references a metric vocabulary and
  BPE model absent from its repository. Restore those exact training assets before
  loading; guessing metric order could silently relabel outputs.
- `vvwangvv/universa-ext_wavlm-base_5metric` uses a separate HyperPyYAML
  `urgent2026_sqa.model.UniVersaExt` architecture. It is not covered by these
  ESPnet state-dict compatibility tests.
- Other collection variants were inspected at config level, not all loaded and
  fine-tuned. ARECHO inference currently supports batch size one.

See the [URGENT24 release guide](../egs2/urgent24/aqa1/README.md) for the full,
audio-only, text-only, and no-reference WavLM packages, their pinned revisions,
and the metric vocabulary saved with those releases.
