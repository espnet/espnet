# Sortformer speaker diarization (offline) — ESPnet3 recipe

End-to-end **offline Sortformer** diarization (NVIDIA, ported to be NeMo-free),
trained on **FastMSS-simulated LibriSpeech meeting mixtures** and evaluated on
**AMI** (mixed-headset) with frame-level **DER**.

Model code lives in `espnet2/diar/sortformer/` and
`espnet2/diar/espnet_sortformer_model.py`; the task is
`espnet3.systems.diar.task.SortformerDiarizationTask`. The architecture
reproduces `nvidia/diar_sortformer_4spk-v1` (FastConformer-L 18×512 with 8×
`dw_striding` subsampling → Transformer 18×192 → 4-speaker sigmoid head),
trained with the hybrid Arrival-Time-Sort + Permutation-Invariant BCE loss.

## Stages

```bash
# 1) Prepare data: build AMI eval cuts (always) and, if configured, simulate
#    LibriSpeech training mixtures with FastMSS.
python run.py --stages data_preparation --training_config conf/training.yaml

# 2) Train from scratch on the simulated mixtures.
python run.py --stages train --training_config conf/training.yaml

# 3) Decode + score DER on AMI dev/test.
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Set the AMI manifest location via `data_prep.ami_dir` in `conf/training.yaml`
(default `/raid/users/popcornell/AMI`). AMI eval windows are rebuilt from
`ami-ihm-mix_recordings_*` + `ami-ihm-mix_supervisions_*` because the shipped
`ami-ihm-mix_cutset_*_30s` manifests carry no supervisions.

## Training data (FastMSS)

Generating the LibriSpeech mixtures needs a checkout of
[FastMSS](https://github.com/popcornell/FastMSS) (branch `librispeech`) plus
LibriSpeech audio + alignments + WHAM noise on disk. Configure the optional
`data_prep.fastmss` block in `conf/training.yaml`, then point
`dataset/config.yaml` `splits.train` at the generated
`synth-librispeech-train-cuts.jsonl.gz`.

## Evaluate NVIDIA's pretrained weights

You can convert and score the released checkpoint without training:

```bash
python -m espnet2.diar.sortformer.convert_hf_sortformer \
    --hf_model nvidia/diar_sortformer_4spk-v1 --out exp/sortformer_4spk.pth
# then set conf/inference.yaml model.model_file: exp/sortformer_4spk.pth
```

The port is numerically faithful: outputs match the original NeMo model to
< 1e-4 (max abs diff) on AMI audio.

## Notes

- Offline (per-window) **and** streaming (speaker-cache, long-form) inference are
  supported; see the long-form section below.
- Fixed 4 speakers (`model.num_spk`).
- The per-window `measure` stage reports frame-level DER (80 ms, Hungarian
  mapping, no collar); the `measure_longform` stage reports collar-based
  session-level DER.

## Long-form (full-session) diarization with the streaming speaker cache

Per-window decoding does not track speaker identity across a meeting. For proper
**session-level DER**, use the streaming speaker cache (one pass per recording,
globally-consistent speakers). Best results use the streaming-trained model
`nvidia/diar_streaming_sortformer_4spk-v2` (dump its weights once in a NeMo env):

```bash
python -c "from nemo.collections.asr.models import SortformerEncLabelModel as M; \
  import torch; m=M.from_pretrained('nvidia/diar_streaming_sortformer_4spk-v2', map_location='cpu'); \
  torch.save(m.state_dict(),'exp/sortformer_v2_full.pt')"

python run.py --stages infer_longform measure_longform \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

Configure the `longform:` blocks in `conf/inference.yaml` (model source, AMI dir,
mic condition `sdm`/`mdm`/`ihm-mix`, split, chunk/overlap) and `conf/metrics.yaml`
(`collar`, default 0.25). DER uses `pyannote.metrics` if installed, else a
frame-level Hungarian fallback (no collar). Results are written to
`${inference_dir}/longform/longform_der.json`.

**Reference result** (converted `diar_streaming_sortformer_4spk-v2`, AMI SDM =
Array1-01 = "array1 mic1", dev, collar 0.25): **DER 19.2%** (miss 11.7 / FA 3.6 /
confusion 3.9). The speaker cache reduces confusion ~7x vs offline chunk-stitching.

## Weight conversion utilities

| Source | Converter |
|---|---|
| `nvidia/diar_sortformer_4spk-v1` (offline, HF) | `python -m espnet2.diar.sortformer.convert_hf_sortformer --out exp/sf_v1.pth` |
| `nvidia/diar_streaming_sortformer_4spk-v2` (.nemo) | dump full state-dict, then `convert_nemo_sortformer.convert_nemo(...)` |
| `nvidia/ssl_en_nest_large_v1.0` (NEST SSL encoder init) | `convert_nest.load_nest_encoder(model, nest_encoder.pt)` |
