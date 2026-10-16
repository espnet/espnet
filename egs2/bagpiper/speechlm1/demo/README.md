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
singing — and `BAGPIPER_MODEL_TAG` switches this page to it.

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
[`egs2/bagpiper_tts/speechlm1/README.md`](https://github.com/espnet/espnet/tree/master/egs2/bagpiper_tts/speechlm1#vllm-serving).

The fork documents audio generation with `espnet/bagpiper-tts-sft`, so
**deploy that converted checkpoint for served rendering**. It documents audio
output from `espnet/bagpiper-sft` as unsupported, because its decoder does
not enforce the text-then-audio turn that checkpoint needs — which is
narrower than that checkpoint being unservable. `BAGPIPER_URL` connects to
whatever the deployment serves: the page neither inspects nor changes the
server's checkpoint, and `BAGPIPER_MODEL_TAG` affects local loading only. To
render with `espnet/bagpiper-sft` — music and environmental sound included —
load it here instead.

## Elsewhere

`espnet describe audio.wav` and `espnet render "A bell rings twice."` are the
same two things from a terminal, and the MCP server offers them as
`describe` and `render`.

## This one is not hosted

There is no `espnet/bagpiper` Space, and this directory is the app rather
than the source of one. Loading the model pulls about 106 GB before it can
answer anything — the 18 GB checkpoint, Qwen3-8B-Base, Xcodec, and
`Qwen/Qwen3-Omni-30B-A3B-Instruct`, all 70.5 GB of which is instantiated so
that its audio tower can be kept and the rest deleted. No free Space
hardware holds that, and ZeroGPU's A10G could not run it anyway: the
published configs ask for FlashAttention-3, which is Hopper-only.

So run it yourself, one of three ways:

- **on a GPU** — `python app.py`, with `ESPNET_BAGPIPER_ATTN=sdpa` on
  anything before Hopper;
- **against a served model** — `BAGPIPER_URL=http://127.0.0.1:9811/v1
  python app.py`, which needs no GPU on this side. Rendering then sends the
  constant audio-generation system prompt, which is what the fork's
  [reference client](https://github.com/espnet/vllm/blob/main/examples/espnet/clients/client_bagpiper.py)
  sends by default and the request it reports validating.
  `ESPNET_BAGPIPER_TTS_SYSTEM=` (empty) sends no system turn instead, which
  is how the Bagpiper-TTS paper describes that checkpoint's training data;
  that variant has not been validated against a served checkpoint, so treat
  it as an override rather than the recipe;
- **from a cluster** — run it inside a batch job and forward the port:
  `ssh -L 7860:<node>:7860 <cluster>`, then open `localhost:7860`.

The demo that is public is the authors' own, at
[bagpiper-cmu.github.io](https://bagpiper-cmu.github.io/). What ESPnet ships
for Bagpiper is `espnet describe` / `espnet render` and the two MCP tools.

If an audio-encoder-only repository ever replaces that 70.5 GB download, a
hosted Space becomes possible again, and this directory is ready for it.
