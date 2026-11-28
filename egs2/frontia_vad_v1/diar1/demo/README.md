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
  - espnet/frontia_vad_v2_3
---

# Voice activity detection

The source of a Hugging Face Space for frontia-vad, kept in this repository
like the other demos. One dual-mode temporal convolutional network (the
model listed in `models:` above, trained as one-speaker diarization,
`num_spk 1`) gives every 8 ms frame the probability that someone is
speaking; a threshold and a median filter turn those into speech segments.
It runs in two modes:

| | model | reads ahead | used for |
|---|---|---|---|
| online | the model at look-ahead 0 | nothing | live streams, the microphone |
| offline | the model at its largest look-ahead | 124 frames (0.99 s) | whole files, cleaning a corpus |

`app.py` loads the model of each mode from `VAD_MODELS` in
`espnet2.diar.vad_models`, so the Space follows the espnet release it
installs. The card above lists the same tags, because Hugging Face reads
them from its front matter; `test_demo_apps.py` checks that the two agree.

The app has three tabs:

- **Recording**: one file, detected in both modes, both drawn on the
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

Each mode has a threshold and a centred median filter, kept as
`VAD_RULES` in `espnet2.diar.vad_models` and tuned on AMI MH dev. The rule is stage 7's (`pyscripts/utils/make_rttm.py`): a
frame is speech when its probability is above the threshold, and the 0/1
sequence is then median-filtered. All four are sliders here, starting at those
values, and `espnet detect` takes `--threshold` and `--median`.

**The default online detection is not strictly causal.** The model reads no
future audio, but a centred median of M frames waits for the (M - 1) / 2
frames after each frame: 96 ms at 25. Offline adds the look-ahead to that.
A median of 1 gives a strictly causal online decision, with no delay.

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

To publish, from the ESPnet checkout, once the model is on the Hub:

```sh
hf auth login
hf upload espnet/frontia-vad egs2/frontia_vad_v1/diar1/demo . --repo-type space
```
