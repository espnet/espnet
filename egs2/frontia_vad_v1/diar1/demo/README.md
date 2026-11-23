---
title: Voice activity detection
emoji: 🎙️
colorFrom: gray
colorTo: blue
sdk: gradio
python_version: "3.12"
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Where is someone speaking? Online and offline VAD
tags:
  - espnet
  - voice-activity-detection
  - vad
  - diarization
models:
  - espnet/frontia_vad_v1_offline
  - espnet/frontia_vad_v1_online
---

# Voice activity detection

The source of a Hugging Face Space for the VAD models of the recipe in the
directory above this one, kept in this repository like the other demos. Two
temporal convolutional networks, each trained as one-speaker diarization
(`num_spk 1`), give every 8 ms frame the probability that someone is
speaking; a threshold and a median filter turn those into speech segments.

| | model | reads ahead | used for |
|---|---|---|---|
| online | causal TCN (v1) | nothing | live streams, the microphone |
| offline | dual-mode TCN, at its largest look-ahead | 124 frames (0.99 s) | whole files, cleaning a corpus |

Their tags are `espnet2.bin.diar_inference.VAD_MODELS`, repeated in `app.py` and in the card
above because a Space is one directory (`test_demo_apps.py` checks that the
three agree).

The app has three tabs:

- **Recording**: one file, detected by both models, both drawn on the
  waveform, with the offline segments as a table.
- **Live**: the microphone, each frame decided by the online model as soon
  as its audio arrives. `espnet2.bin.diar_inference_streaming.StreamingVAD` keeps the causal
  encoder's past and runs only the new frames, so the stream gives the
  probabilities the whole recording would.
- **Many files**: up to 20 recordings detected by the offline model in one
  batch through `DiarizeSpeech.diarize_batch`, as for cleaning a corpus.

The detection is `espnet2.bin.diar_inference` (offline, batches) and
`espnet2.bin.diar_inference_streaming` (online), which `espnet detect` and the MCP tool
`detect` also call, so the three agree on a file.

## The decision rules

Each model has its own threshold and median filter: the setting with the
lowest detection error on AMI MH dev (`../local/show_vad_result.py`), kept as
`espnet2.bin.diar_inference.VAD_RULES`. The rule is stage 7's
(`pyscripts/utils/make_rttm.py`): a frame is speech when its probability is
above the threshold, and the 0/1 sequence is then median-filtered. All four
are sliders here, starting at the tuned values.

## Running and publishing

```sh
pip install -r requirements.txt gradio
VAD_MODEL_TAG=/path/to/offline.zip VAD_ONLINE_MODEL_TAG=/path/to/online.zip \
    python app.py   # http://127.0.0.1:7860
```

`VAD_MODEL_TAG` (offline) and `VAD_ONLINE_MODEL_TAG` (online) take a Hugging
Face tag or the `.zip` that stage 8 of `diar.sh` packs. `DEVICE` decides
where the models run and takes precedence over everything else; with it unset
the choice is CUDA on ZeroGPU (`SPACES_ZERO_GPU`) or when torch reports a
GPU, and CPU otherwise. The models are a few hundred kilobytes each and run
on the free CPU tier.

Each recording, and a live session, is cut at 120 s (`MAX_SECS`), with a
warning saying so; `espnet detect` has no such limit.

`requirements.txt` asks for the first espnet release with the VAD classes
and both TCN encoders; until that release is out the Space cannot be built.

To publish, from the ESPnet checkout, once both models are on the Hub:

```sh
hf auth login
hf upload espnet/frontia-vad egs2/frontia_vad_v1/diar1/demo . --repo-type space
```
