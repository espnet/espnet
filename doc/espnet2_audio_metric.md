# Audio metric prediction: Uni-VERSA and ARECHO

The `audio_metric` task revives the regression models from
[PR #5959](https://github.com/espnet/espnet/pull/5959) and the autoregressive
ARECHO model from `ftshijt/espnet`'s `universa_plus` branch. The task name describes
the function; model names and parameter prefixes retain `universa` for checkpoint
compatibility. See the [recipe guide](../egs2/TEMPLATE/audio_metric1/README.md)
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

Old `UniversaTask`, `espnet2.bin.universa_train`, and
`espnet2.bin.universa_inference` imports are aliases. Old recipe directory and
`uni_versa.sh` names are symlinks. This preserves model and entry-point loading;
it does not promise every experimental application in `arecho_app` is supported.

Use the original architecture config with resolved artifact paths and
`--init_param /path/to/model.pth` to initialize fine-tuning. Preserve `freeze_param`
to freeze the same modules as the released config. An epoch weight file alone
cannot restore optimizer/scheduler state; `--resume true` needs a full trainer
checkpoint. New training intentionally uses the corrected positive objective.

## Published checkpoint validation

The following representative weights were strictly loaded (no missing/unexpected
keys), then trained for one optimizer step with their configured frozen parameter
prefixes. The check requires finite loss and gradients, verifies a trainable
parameter changed, and runs inference afterward.

| Model | Hugging Face revision | Weight |
| --- | --- | --- |
| [Uni-VERSA WavLM no-reference](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_noref) | `6abec38c0971c2c0feebbe39ee2f94c9c21357c4` | `update_exp/universa_train_universa_wavlm_noref_raw_fs16000/13epoch.pth` |
| [ARECHO base v0](https://huggingface.co/espnet/arecho_base_v0) | `e6619474fc7b462db8da13183c9b565778c07980` | `exp/universa_universa_ar_overall_base_token_wavlm/68epoch.pth` |

Run the opt-in check from the repository root after downloading the model package
at the recorded revision and preparing the WavLM cache:

```bash
PYTHONPATH=. python test/espnet2/universa/check_published_checkpoint.py \
  /path/to/arecho_base_v0 \
  --config exp/universa_universa_ar_overall_base_token_wavlm/config.yaml \
  --checkpoint exp/universa_universa_ar_overall_base_token_wavlm/68epoch.pth \
  --frontend-cache /path/to/wavlm-cache \
  --metrics '{"srmr": 2.5, "language": "<eng>"}'
```

For the Uni-VERSA package, use its paths from the table and `--metrics '{"mos": 2.5}'`.
The check reads local artifacts; it does not download model packages. The frontend
library may fetch its upstream model if the supplied cache is incomplete.

Validation used CPU PyTorch 2.11 and the available Python 3.10 environment, with
current `espnet-s3prl`. This is a compatibility smoke check, not a quality benchmark
or validation of every checkpoint. Current ESPnet's supported Python 3.12/3.13
matrix and GPU runs remain CI work. A separate tiny model completed ESPnet
statistics collection and one epoch of training. Full shell recipes were syntax
checked but not executed locally because sph2pipe was unavailable.

Artifact gaps found in the supplied collections:

- `espnet/universa-base_urgent24_multi-metric` references a metric vocabulary and
  BPE model absent from its repository. Restore those exact training assets before
  loading; guessing metric order could silently relabel outputs.
- `vvwangvv/universa-ext_wavlm-base_5metric` uses a separate HyperPyYAML
  `urgent2026_sqa.model.UniVersaExt` architecture. It is not covered by these
  ESPnet state-dict compatibility tests.
- Other collection variants were inspected at config level, not all loaded and
  fine-tuned. ARECHO inference currently supports batch size one.
