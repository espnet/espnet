---
title: OWSM-CTC v4
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 151 languages in, 25 translation targets out, fast
tags:
  - espnet
  - owsm
  - automatic-speech-recognition
  - speech-translation
  - language-identification
models:
  - espnet/owsm_ctc_v4_1B
---

# OWSM-CTC v4

The source of [`espnet/owsm-ctc-v4`](https://huggingface.co/spaces/espnet/owsm-ctc-v4),
one of two OWSM v4 demos kept in this repository. Both offer the same tasks -
speech recognition, any-to-any speech translation, language identification and
long-form decoding - so the two models can be compared on the same audio:

| | this demo | `espnet/owsm-v4` |
|---|---|---|
| model | [`owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B), encoder-only | [`owsm_v4_medium_1B`](https://huggingface.co/espnet/owsm_v4_medium_1B), encoder-decoder |
| decoding | one encoder pass per 30 s window, no beam search | beam search |
| text prompt | not supported by OWSM-CTC | supported |
| source | this directory | [`egs2/owsm_v4/s2t1/demo`](../../../owsm_v4/s2t1/demo) |

The language menu and the translation targets are read from the checkpoint, so
a model covering more languages needs no edit here.

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
hf upload espnet/owsm-ctc-v4 egs2/owsm_ctc_v4/s2t1/demo . --repo-type space
```

## Citation

```bibtex
@inproceedings{owsm-ctc,
  title={{OWSM-CTC}: An Open Encoder-Only Speech Foundation Model for Speech
         Recognition, Translation, and Language Identification},
  author={Yifan Peng and Yui Sudo and Muhammad Shakeel and Shinji Watanabe},
  booktitle={Proc. ACL},
  year={2024}
}
```
