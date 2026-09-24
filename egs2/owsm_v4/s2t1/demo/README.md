---
title: OWSM v4
emoji: 🌍
colorFrom: blue
colorTo: pink
sdk: gradio
python_version: "3.12"
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

The source of [`espnet/owsm-v4`](https://huggingface.co/spaces/espnet/owsm-v4),
one of two OWSM v4 demos kept in this
repository. Both offer the same tasks - speech recognition, any-to-any speech
translation, language identification and long-form decoding - so the two models
can be compared on the same audio:

| | this demo | [`espnet/owsm-ctc-v4`](https://huggingface.co/spaces/espnet/owsm-ctc-v4) |
|---|---|---|
| model | [`owsm_v4_medium_1B`](https://huggingface.co/espnet/owsm_v4_medium_1B), encoder-decoder | [`owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B), encoder-only |
| decoding | beam search | one encoder pass per 30 s window |
| text prompt | supported | not supported by OWSM-CTC |
| source | this directory | [`egs2/owsm_ctc_v4/s2t1/demo`](../../../owsm_ctc_v4/s2t1/demo) |

## The page is not here

`app.py` loads a model, picks a device and asks for a slice of GPU time. The
page is [`espnet2.bin.demo`](../../../../espnet2/bin/demo.py), the module
`espnet demo` serves, which reads what to offer off the checkpoint - the
language menu, the translation targets, the window, and the text prompt,
which appears here because this model has a decoder to prime and not on the
encoder-only one, which has nothing to prime.

Both demos used to carry their own copy of that page.
`test/espnet2/bin/test_demo_apps.py` existed to keep them in step; the copies
are gone.

Language identification uses `Speech2Language` on the same checkpoint; the
language menu and the translation targets are read from it too.

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
