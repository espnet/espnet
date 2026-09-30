---
title: LJSpeech VITS
emoji: 🗣️
colorFrom: purple
colorTo: pink
sdk: gradio
python_version: "3.12"
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: English text to speech with a VITS model
tags:
  - espnet
  - tts
  - text-to-speech
  - vits
models:
  - espnet/kan-bayashi_ljspeech_vits
---

# LJSpeech VITS

The source of the Hugging Face Space `espnet/ljspeech-vits`, one of the demos
kept in this repository. It types text in and gets a waveform out:
[`kan-bayashi_ljspeech_vits`](https://huggingface.co/espnet/kan-bayashi_ljspeech_vits)
is an end-to-end model with no separate vocoder, trained on LJSpeech by the
recipe in the directory above this one. It is the model `espnet synthesize` uses by
default.

The controls are read from the checkpoint rather than written here:

| control | shown when | LJSpeech VITS |
|---|---|---|
| Speaker | the model takes speaker IDs (`tts.spks`) | hidden: one speaker |
| Language | the model takes language IDs (`tts.langs`) | hidden: English only |
| Speed | the model's decoding config has `alpha` | shown: VITS scales durations |

So `TTS_MODEL_TAG` can point the same app at a multi-speaker or multi-lingual
checkpoint and the menus fill themselves. A checkpoint that needs a speaker
embedding or a reference recording — an x-vector or GST model — is refused at
startup, with the reason, rather than failing inside the first request.

The hosted demo reads up to 500 characters and asks ZeroGPU for a matching
slice of GPU time; text beyond that is cut, with a warning saying so. The
model itself has no such limit, so run the app yourself for more.

The card pins `python_version: "3.12"`. espnet requires 3.12 or 3.13, and a
Space image with an older interpreter installs no espnet at all - pip reports
"No matching distribution found" and the build fails.

`requirements.txt` asks for `espnet[tts]`, not plain `espnet`: this
checkpoint's tokenizer is `g2p_en_no_space` and its cleaner is `tacotron`,
and both live in that extra. Without it the model loads and the first
request fails.

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
hf upload espnet/ljspeech-vits egs2/ljspeech/tts1/demo . --repo-type space
```

The Space itself has to exist as ZeroGPU hardware (`zero-a10g`), which is a
setting on the Space rather than something in these files.

## Citation

```bibtex
@article{hayashi2021espnet2,
  title={{ESP}net2-{TTS}: Extending the edge of {TTS} research},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Yoshimura, Takenori and
          Wu, Peter and Shi, Jiatong and Saeki, Takaaki and Ju, Yooncheol and
          Yasuda, Yusuke and Takamichi, Shinnosuke and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2110.07840},
  year={2021}
}
```
