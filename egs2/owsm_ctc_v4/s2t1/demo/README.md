---
title: OWSM-CTC v4
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.12"
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

## The page is not here

`app.py` loads a model, picks a device and asks for a slice of GPU time. The
page is [`espnet2.bin.demo`](../../../../espnet2/bin/demo.py), the module
`espnet demo` serves, which reads what to offer off the checkpoint: the
language menu, the translation targets, the decoding window, and whether
there is a decoder to prime with a text prompt. So `python app.py` and
`espnet demo --model espnet/owsm_ctc_v4_1B` are the same page, and a fix to
either is a fix to both.

This app used to carry its own copy of that page, and so did
`egs2/owsm_v4/s2t1/demo`; `test/espnet2/bin/test_demo_apps.py` existed to
keep the three in step. The copies are gone.

The hosted demo takes audio of up to two minutes and asks ZeroGPU for a
matching slice of GPU time; the model itself has no such limit, so run the app
yourself for longer recordings.

The card pins `python_version: "3.12"`. espnet requires 3.12 or 3.13, and a
Space image with an older interpreter installs no espnet at all - pip reports
"No matching distribution found" and the build fails.

## Running and publishing

```sh
pip install -r requirements.txt gradio
python app.py                     # http://127.0.0.1:7860
```

`DEVICE` decides where the models run and takes precedence over everything
else, so `DEVICE=cpu` is a local CPU test even on a machine with a GPU. With
`DEVICE` unset, the choice is CUDA when the runtime is ZeroGPU
(`SPACES_ZERO_GPU`, the marker Hugging Face sets — asking
`torch.cuda.is_available()` there answers False, because the GPU is attached
only while a `@spaces.GPU` function runs) or when torch reports a GPU, and CPU
otherwise.

To publish, from the ESPnet checkout:

```sh
hf auth login
hf upload espnet/owsm-ctc-v4 egs2/owsm_ctc_v4/s2t1/demo . --repo-type space
```

**Not before the release named in `requirements.txt`.** A Space installs
espnet from PyPI, so an app that imports something only master has is a Space
that builds and then fails to start. This app imports `espnet2.bin.demo`,
whose `build_app` took its `wrap` argument in 202610.post2 - hence
`espnet[demo]>=202610.post2`, which pip will refuse until that release
exists. Uploading this directory before then replaces a working Space with a
broken one, which is exactly what happened on 2026-09-20.

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
