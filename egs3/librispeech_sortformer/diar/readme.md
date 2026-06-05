# Sortformer speaker diarization — ESPnet3 (main recipe)

The **main ESPnet3 diarization recipe**: an **8-speaker streaming Sortformer**
(NVIDIA Sortformer ported NeMo-free), with the **speaker cache** in the loop
(original-Sortformer style), the FastConformer **initialized from NEST**
self-supervised weights, trained on **FastMSS** 1-minute, **3–8-speaker**
simulated LibriSpeech meetings **+ AMI** single-distant-mic, and evaluated with
**long-form (full-session) DER on AMI**.

Model/library code is under `espnet2/diar/sortformer/` and
`espnet2/diar/espnet_sortformer_model.py`; the task is
`espnet3.systems.diar.task.SortformerDiarizationTask`. The 8-speaker loss uses
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
- `fastmss.librispeech_dir` / `fastmss.librispeech_align` — LibriSpeech audio and
  **forced alignments**. ⚠️ FastMSS uses `lhotse.recipes.prepare_librispeech`,
  which expects the **LibriSpeech-Alignments `.txt`** format (not MFA
  `.TextGrid`). If your alignments are TextGrids, convert them or point at the
  `.txt`-format release, otherwise forced-alignment splitting yields 0 cuts.
- `fastmss.noise_folders` — a noise directory (WHAM/MUSAN, 16 kHz). Set this; if
  omitted, generation runs reverb-only (no additive noise).
- `min_max_spk: [3, 8]`, `duration: 60` (1-minute meetings).

AMI cuts are built from `ami-sdm_{recordings,supervisions}_{train,dev,test}` in
`data_prep.ami_dir` via `cut_into_windows(60)` (SDM = `Array1-01` = array1 mic1).
Training data = FastMSS meetings **+** AMI SDM train (combined by the
`DataOrganizer`); validation = AMI SDM dev.

## Model

FastConformer = NEST-L architecture (18×512, 80-mel, dw_striding 8×), initialized
from NEST; Transformer (18×192) + 8-speaker sigmoid head trained from scratch.
Streaming speaker cache (`spkcache_len`/`chunk_len` 188, `fifo_len` 0); training
runs the cache in the loop (`model_conf.train_streaming: true`) so long-form
inference keeps speaker identity globally consistent.

## Evaluation

`infer_longform` runs full-session streaming inference (one pass per recording
with the speaker cache) → one RTTM/session; `measure_longform` scores collar-
based DER with `pyannote.metrics` (frame-level fallback if absent) →
`${inference_dir}/longform/longform_der.json`.

## Variants / utilities

- `conf/training_4spk_offline.yaml` — legacy 4-speaker **offline** model
  (architecture of `nvidia/diar_sortformer_4spk-v1`).
- Weight conversion: `espnet2/diar/sortformer/convert_hf_sortformer.py`
  (offline v1 HF), `convert_nemo_sortformer.py` (streaming v2 `.nemo`),
  `convert_nest.py` (NEST encoder init). The offline port is numerically faithful
  (parity < 1e-4 vs the original on AMI audio).
