---
title: Universal speech enhancement
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: "3.12"
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: One model for noise, reverb, any mics, any rate
tags:
  - espnet
  - speech-enhancement
  - audio-to-audio
  - uses
models:
  - espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw
---

# Universal speech enhancement

The source of the Hugging Face Space `espnet/universal-se`, one of the demos
kept in this repository. USES is a single enhancement model for every
condition the task usually splits into separate ones — noise, reverberation,
one microphone or several, and any sample rate — trained by the recipe in the
directory above this one. It is the model `espnet enhance` uses by default.

What the app reads from the checkpoint rather than from this file:

| from the checkpoint | how the app uses it |
|---|---|
| the training categories (`1ch_48k`, `2ch_16k`, …) | the rates in the "Process at" menu |
| the largest channel count those categories name | a warning when a file has more |
| `num_spk` | one output player per speaker, so a separation checkpoint reached through `ENH_MODEL_TAG` grows the column instead of dropping speakers |

Multi-channel files keep their channels on the way in, because a model that
can use them should be given them; the model returns one channel.

The hosted demo enhances up to 30 seconds and asks ZeroGPU for a matching
slice of GPU time; longer audio is trimmed, with a warning saying so. The
model itself has no such limit, so run the app yourself for longer
recordings.

The card pins `python_version: "3.12"`. espnet requires 3.12 or 3.13, and a
Space image with an older interpreter installs no espnet at all - pip reports
"No matching distribution found" and the build fails.

`requirements.txt` asks for `espnet[enh]`, not plain `espnet`: building this
model builds the training criteria with it, and the SI-SNR loss imports
`fast-bss-eval`. Without the extra the model does not load at all.

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
hf upload espnet/universal-se egs2/universal_se_v1/enh1/demo . --repo-type space
```

The Space itself has to exist as ZeroGPU hardware (`zero-a10g`), which is a
setting on the Space rather than something in these files. This model is the
slowest of the five demos on a CPU — around three times real time — so the
GPU is what makes it usable.

## Citation

```bibtex
@inproceedings{zhang2023uses,
  title={Toward Universal Speech Enhancement for Diverse Input Conditions},
  author={Zhang, Wangyou and Saijo, Kohei and Wang, Zhong-Qiu and
          Watanabe, Shinji and Qian, Yanmin},
  booktitle={Proc. ASRU},
  year={2023}
}
```
