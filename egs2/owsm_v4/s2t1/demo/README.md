---
title: OWSM v4
emoji: 🌍
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 151 languages in, 25 translation targets out, promptable
tags:
  - espnet
  - owsm
  - automatic-speech-recognition
  - speech-translation
  - language-identification
models:
  - espnet/owsm_v4_medium_1B
---

# OWSM v4

The source of `espnet/owsm-v4`, one of two OWSM v4 demos kept in this
repository. Both offer the same tasks - speech recognition, any-to-any speech
translation, language identification and long-form decoding - so the two models
can be compared on the same audio:

| | this demo | [`espnet/owsm-ctc-v4`](https://huggingface.co/spaces/espnet/owsm-ctc-v4) |
|---|---|---|
| model | [`owsm_v4_medium_1B`](https://huggingface.co/espnet/owsm_v4_medium_1B), encoder-decoder | [`owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B), encoder-only |
| decoding | beam search | one encoder pass per 30 s window |
| text prompt | supported | not supported by OWSM-CTC |
| source | this directory | [`egs2/owsm_ctc_v4/s2t1/demo`](../../../owsm_ctc_v4/s2t1/demo) |

Language identification uses `Speech2Language` on the same checkpoint; the
language menu and the translation targets are read from it too.

The hosted demo takes audio of up to two minutes and asks ZeroGPU for a
matching slice of GPU time; the model itself has no such limit, so run the app
yourself for longer recordings.

## Running and publishing

```sh
pip install -r requirements.txt gradio
python app.py                     # http://127.0.0.1:7860
```

`DEVICE` chooses where it runs; on the Space it is the GPU that ZeroGPU
attaches to the decorated function. To publish, from the ESPnet checkout:

```sh
hf auth login
hf upload espnet/owsm-v4 egs2/owsm_v4/s2t1/demo . --repo-type space
```

The Space itself has to exist as ZeroGPU hardware (`zero-a10g`), which is a
setting on the Space rather than something in these files.

## Citation

```bibtex
@inproceedings{owsm-v4,
  title={{OWSM} v4: Improving Open Whisper-Style Speech Models via Data Scaling
         and Cleaning},
  author={Yifan Peng and Shakeel Muhammad and Yui Sudo and William Chen and
          Jinchuan Tian and Chyi-Jiunn Lin and Shinji Watanabe},
  booktitle={Proc. Interspeech},
  year={2025}
}
```
