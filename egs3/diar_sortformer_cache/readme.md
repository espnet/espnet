# Sortformer speaker diarization — ESPnet3 (main recipe)

The **main ESPnet3 diarization recipe**: an **8-speaker streaming Sortformer**
(NVIDIA Sortformer ported NeMo-free), with the **speaker cache** in the loop, the
FastConformer **initialized from NEST** self-supervised weights, trained on
**FastMSS** **3–8-speaker** simulated LibriSpeech meetings **+ AMI**
single-distant-mic, and evaluated with **long-form (full-session) DER on AMI**.

Streaming hyper-parameters follow NeMo's released streaming-v2 config
(`diar_streaming_sortformer_4spk-v2`): **90 s sessions**, chunk/cache **188**
(~15 s), `fifo_len` 0, `sil_frames` 3, full attention. The `causal_attn_rate`
augmentation is intentionally omitted (this recipe does not target causal /
low-latency streaming). Intended departures from v2 are **8 speakers** (v2 = 4)
and **NEST initialization** (80-mel / 18-layer; v2 = 128-mel / 17-layer).

Model/library code is under `sortformer/` and
`sortformer/model.py`; the task is
`sortformer.task.SortformerDiarizationTask`. The 8-speaker loss uses
**Hungarian PIL + arrival-time-argsort ATS** (brute-force `S!` is infeasible at
8 speakers).

## Stages

```bash
# 0) (once, in a NeMo env) dump the NEST encoder weights for initialization:
python -c "from nemo.collections.asr.models import ASRModel; import torch; \
  m=ASRModel.from_pretrained('nvidia/ssl_en_nest_large_v1.0', map_location='cpu'); \
  torch.save(m.encoder.state_dict(), 'exp/nest_encoder.pt')"

# 1) data prep: generate FastMSS meetings + build AMI SDM cuts
python run.py --stages data_preparation --training_config conf/training.yaml

# 2) train the 8-spk streaming Sortformer (speaker cache in the loop)
python run.py --stages train --training_config conf/training.yaml

# 3) long-form (full-session) diarization + collar DER on AMI
python run.py --stages infer_longform measure_longform \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`python run.py` (no `--stages`) runs the full chain
`data_preparation → train → infer_longform → measure_longform`.

## Data preparation (stage 1)

Edit the `data_prep` block in `conf/training.yaml`:

- `fastmss.dir` — a checkout of [FastMSS](https://github.com/popcornell/FastMSS)
  **`librispeech` branch** (e.g. `downloads/FastMSS`).
- `fastmss.aligned_manifests` (**preferred**) — a directory of **word-aligned
  lhotse LibriSpeech manifests** (`librispeech_{recordings,supervisions}_<split>.jsonl.gz`
  whose supervisions carry `alignment["word"]`). When set, FastMSS skips its
  stage-0 alignment download and uses these directly (the recipe pre-builds
  `all_cuts_orig.jsonl.gz` and starts FastMSS at stage 1).
- `fastmss.librispeech_align` (fallback, only if `aligned_manifests` is unset) —
  `auto` downloads the lhotse-format `.txt` alignments via lhotse
  `download_librispeech`, or point it at a `.../LibriSpeech-Alignments/LibriSpeech`
  dir. The on-disk MFA `.TextGrid` alignments are NOT compatible with
  `lhotse.prepare_librispeech`.
- `fastmss.noise_folders` — a noise directory (WHAM/MUSAN, 16 kHz; short clips are
  looped). If omitted, generation runs reverb-only.
- `min_max_spk: [3, 8]`, `duration: 90` (90 s meetings).

AMI cuts are built from `ami-sdm_{recordings,supervisions}_{train,dev,test}` in
`data_prep.ami_dir` via `cut_into_windows(90)` (SDM = `Array1-01` = array1 mic1).
Training data = FastMSS meetings **+** AMI SDM train (combined by the
`DataOrganizer`); validation = AMI SDM dev.

## Model

FastConformer = NEST-L architecture (18×512, 80-mel, dw_striding 8×), initialized
from NEST; Transformer (18×192) + 8-speaker sigmoid head trained from scratch.
Streaming speaker cache (`spkcache_len`/`chunk_len` 188, `fifo_len` 0,
`sil_frames` 3); training runs the cache in the loop (`model_conf.train_streaming:
true`) over ~6 chunks per 90 s session, so the cache learns to compress/evict and
keep speaker identity globally consistent for long-form inference. AdamW lr 1e-4,
betas (0.9, 0.98), wd 1e-3, warmup 500 (NeMo streaming-v2).

### Efficient local attention (optional)

`encoder_conf.att_context_size` / `transformer_conf.att_context_size` enable an
**O(N·W) chunked sliding-window attention** (Longformer-style band kernel, the
mechanism Parakeet-v3 uses) with the speaker-cache prefix kept global. Set both to
`[left, right]` (80-ms frames) to make the whole model O(N·W) for single-pass
long-form; `null` (default) = full attention (NeMo-faithful). See
`sortformer/sliding_window_attention.py`.

## Evaluation

`infer_longform` runs full-session streaming inference (one pass per recording
with the speaker cache) → one RTTM/session; `measure_longform` scores collar-based
DER with `pyannote.metrics` (frame-level fallback if absent) →
`${inference_dir}/longform/longform_der.json`.

## Variants / utilities

- `conf/training_4spk_offline.yaml` — legacy 4-speaker **offline** model
  (architecture of `nvidia/diar_sortformer_4spk-v1`).
- Weight conversion: `sortformer/convert_hf_sortformer.py`
  (offline v1 HF), `convert_nemo_sortformer.py` (streaming v2 `.nemo`),
  `convert_nest.py` (NEST encoder init). The offline port is numerically faithful
  (parity < 1e-4 vs the original on AMI audio); converting the released streaming
  v2 reaches ~19% long-form DER on AMI SDM.
