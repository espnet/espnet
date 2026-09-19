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
short_description: One model for noise, reverb, and 8-48 kHz
tags:
  - espnet
  - speech-enhancement
  - audio-to-audio
  - tfgridnet
models:
  - kohei0209/tfgridnet_urgent25
---

# Universal speech enhancement

The source of the Hugging Face Space `espnet/universal-se`, one of the demos
kept in this repository. It runs the TF-GridNet baseline of the
[URGENT 2025 challenge](https://urgent-challenge.github.io/urgent2025), a
single enhancement model for noise, reverberation and every sample rate from
8 to 48 kHz, and it is the model `espnet enhance` uses by default.

The app reads everything model-specific off the checkpoint, so `ENH_MODEL_TAG`
points it at another one — the USES model of the recipe in the directory above
this one, say, which is what this directory was written around. The URGENT 2025
recipe that trained the default is not upstream yet; when it lands, this demo
moves next to it.

What the app reads from the checkpoint rather than from this file:

| from the checkpoint | how the app uses it |
|---|---|
| the training categories (`1ch_16000Hz`, `2ch_16k`, …; recipes spell the rate either way) | the rates in the "Process at" menu |
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
slowest of the five demos on a CPU — measured on one here, about three times
real time at 16 kHz and nine times at 48 kHz — so the GPU is what makes it
usable.

## Citation

```bibtex
@inproceedings{li2020espnetse,
  title={ESPnet-SE: End-to-End Speech Enhancement and Separation Toolkit
         Designed for ASR Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and
          Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and
          Moto Hira and Tomoki Hayashi and Christoph Boeddeker and
          Zhuo Chen and Shinji Watanabe},
  booktitle={SLT},
  year={2021}
}
```

The checkpoint is released under CC-BY-4.0; see
[its model card](https://huggingface.co/kohei0209/tfgridnet_urgent25).
