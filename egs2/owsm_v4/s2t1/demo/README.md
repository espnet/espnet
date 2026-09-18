---
title: OWSM V4 Demo
emoji: 🌍
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: A demo of the OWSM v4 CTC and medium models
tags:
  - espnet
  - owsm
  - automatic-speech-recognition
  - speech-translation
  - language-identification
models:
  - espnet/owsm_v4_medium_1B
  - espnet/owsm_ctc_v4_1B
---

# OWSM v4 Demo

The source of [`espnet/OWSM_V4_Demo`](https://huggingface.co/spaces/espnet/OWSM_V4_Demo),
which serves both OWSM v4 models: the encoder-decoder
[`owsm_v4_medium_1B`](https://huggingface.co/espnet/owsm_v4_medium_1B) with beam
search, and the encoder-only
[`owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B). It recognises
151 languages, translates into 25, identifies the spoken language, predicts
utterance timestamps and decodes long-form audio.

The Space was maintained on the Hub alone until now; keeping the source here
means it is reviewed, versioned and checked with the rest of the repository —
`ci/check_demo_links.py` verifies this card daily against the rules the Hub
enforces, and reports the Space when it stops running.

The smaller single-model demo lives in
[`egs2/owsm_ctc_v4/s2t1/demo`](../../../owsm_ctc_v4/s2t1/demo) and publishes to
[`espnet/owsm-ctc-v4`](https://huggingface.co/spaces/espnet/owsm-ctc-v4).

## Running and publishing

```sh
pip install -r requirements.txt gradio
python app.py                     # http://127.0.0.1:7860
```

`DEVICE` selects where the models run; it is CUDA on the Space and whatever is
available locally. To publish, from the ESPnet checkout:

```sh
hf auth login
hf upload espnet/OWSM_V4_Demo egs2/owsm_v4/s2t1/demo . --repo-type space
```

## Citation

```bibtex
@inproceedings{owsm-v4,
  title={{OWSM} v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning},
  author={Yifan Peng and Shakeel Muhammad and Yui Sudo and William Chen and Jinchuan Tian and Chyi-Jiunn Lin and Shinji Watanabe},
  booktitle={Proc. Interspeech},
  year={2025}
}
```
