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
short_description: ASR, speech translation and language ID in one encoder
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

[OWSM-CTC](https://aclanthology.org/2024.acl-long.549/) is an encoder-only,
non-autoregressive speech foundation model trained on 320k hours of public
audio. One model transcribes 150 languages, translates speech into 25 of
them and identifies the spoken language, with no beam search: one encoder
pass per 30-second window. It runs without a GPU — about 30 s per request
on a laptop CPU (Apple M1 Pro), longer on a basic CPU Space — and is
interactive on a GPU Space with `DEVICE=cuda`.

Record or upload audio of any length or sample rate, leave the language on
`auto` or pick one, and choose *transcribe* or a translation target. The
model is [`espnet/owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B)
(CC-BY-4.0), run through [ESPnet](https://github.com/espnet/espnet):

```python
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

s2t = Speech2TextGreedySearch.from_pretrained("espnet/owsm_ctc_v4_1B")
print(s2t.batch_decode("audio.wav", lang_sym="<eng>", task_sym="<asr>"))
```

## Running and publishing this demo

The source lives in the ESPnet repository at
[`egs2/owsm_ctc_v4/s2t1/demo`](https://github.com/espnet/espnet/tree/master/egs2/owsm_ctc_v4/s2t1/demo).

```sh
pip install "espnet>=202609.post2" gradio
python app.py                     # http://127.0.0.1:7860
```

To publish it as a Hugging Face Space, from the ESPnet checkout:

```sh
hf auth login
hf upload espnet/owsm-ctc-v4 egs2/owsm_ctc_v4/s2t1/demo . --repo-type space
```

`OWSM_MODEL_TAG` selects another OWSM-CTC checkpoint. On a GPU Space set
`DEVICE=cuda` in the Space's variables; every request then takes about a
second instead of minutes.

## Citation

```bibtex
@inproceedings{peng2025owsmv4,
  title={{OWSM} v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning},
  author={Yifan Peng and Shakeel Muhammad and Yui Sudo and William Chen and Jinchuan Tian and Chyi-Jiunn Lin and Shinji Watanabe},
  booktitle={Proc. Interspeech},
  year={2025}
}
@inproceedings{peng2024owsmctc,
  title={{OWSM-CTC}: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification},
  author={Yifan Peng and Yui Sudo and Muhammad Shakeel and Shinji Watanabe},
  booktitle={Proc. ACL},
  year={2024}
}
```
