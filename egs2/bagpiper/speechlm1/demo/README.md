---
title: Bagpiper
emoji: 🎻
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - espnet/bagpiper-sft
short_description: Describe a sound, or render one you describe
---

# Bagpiper

Audio understanding and audio generation in one model, joined by the rich
caption it writes in both directions. Ask what a recording is; render what
you describe; or do both in a loop, which is what this page is laid out for.

The model is [`espnet/bagpiper-sft`](https://huggingface.co/espnet/bagpiper-sft),
8B, from [Bagpiper: Solving Open-Ended Audio Tasks via Rich
Captions](https://arxiv.org/abs/2602.05220).
[`espnet/bagpiper-tts-sft`](https://huggingface.co/espnet/bagpiper-tts-sft) is
the speech-focused sibling — multi-talker, intent-to-speech, role-play,
singing — and `BAGPIPER_MODEL_TAG` switches this Space to it.

## Run it yourself

```bash
pip install -r requirements.txt
python app.py
```

It needs a large GPU: 8B weights in bfloat16, plus an audio encoder and a
codec. Set `BAGPIPER_URL=http://127.0.0.1:9811/v1` to point the page at a
model already served by the [ESPnet vLLM
fork](https://github.com/espnet/vllm) instead of loading one here — the
serving command is in
[`egs2/bagpiper/speechlm1/README.md`](https://github.com/espnet/espnet/tree/master/egs2/bagpiper/speechlm1).

## Elsewhere

`espnet describe audio.wav` and `espnet render "A bell rings twice."` are the
same two things from a terminal, and the MCP server offers them as
`describe` and `render`.

## When this may be uploaded

After 202610.post3, the release that carries
`espnet2.bin.speechlm_inference`. Uploading before it exists gives a Space
that builds and then fails to start.
