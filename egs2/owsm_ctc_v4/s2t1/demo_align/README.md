---
title: Forced alignment
emoji: 📐
colorFrom: yellow
colorTo: red
sdk: gradio
python_version: "3.12"
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: When each line was said, and how sure the model is
tags:
  - espnet
  - owsm
  - forced-alignment
  - automatic-speech-recognition
models:
  - espnet/owsm_ctc_v4_1B
---

# Forced alignment

The source of [`espnet/forced-alignment`](https://huggingface.co/spaces/espnet/forced-alignment).
Audio and the text that goes with it in; a start, an end and a score for every
line out, with the waveform marked.

Nothing is trained for this. The alignment is a Viterbi path through the CTC
head of a model that already exists — [OWSM-CTC
v4](https://huggingface.co/espnet/owsm_ctc_v4_1B) here — so it works in any
language that model covers, and `ALIGN_MODEL_TAG` points it at another
checkpoint without touching this file. The sample rate comes from that
checkpoint too (`ForcedAligner.sample_rate`), so a model trained at another
rate needs no edit here either.

## Why this one has a page of its own

The other demos in this repository are one input and one output, and their
page is built from the checkpoint by
[`espnet2.bin.demo`](../../../../espnet2/bin/demo.py): the menus, the window
and the tasks are read off the model, which is how
`egs2/powsm_ctc/s2t1/demo` needs no interface at all.

Alignment does not fit that shape. It takes audio **and** text, and answers
with a table rather than a sentence — so the page is here, and it is the only
demo that draws its own.

## What the score is for

`score` is the mean probability of a line's tokens: 1.0 is a perfect match,
and a line that does not belong to this audio scores near zero. That is what
alignment-score filtering uses — OWSM v4 cleaned its training data by aligning
it and dropping the low scores
([recipe](../../../owsm_v4/s2t1)).

The app warns when the weakest line falls under `ALIGN_WARN_BELOW`, which is
0.3 by default. That number is a **heuristic, not a calibrated confidence**:
it is where a line that was not said sits on OWSM-CTC v4 and read English,
measured on the example this Space ships with (0.000 for a line that does not
belong, against 0.99, 0.77 and 0.99 for the three that do). A checkpoint that
scores differently wants a different number, and the environment variable is
there for that.

It is a probability **under this model**, so the text has to be written the
way the model writes it. Measured on `test_utils/ctc_align_test.wav` with
`owsm_ctc_v4_1B`: "The sale of the hotels" scores 0.99 and the same words in
capitals score 0.0000, because the vocabulary has no capitalised words and
each one breaks into letters. The times stay roughly right either way, which
is what makes it confusing rather than obviously wrong — so the app warns when
the weakest line scores below 0.3, and `espnet2.bin.align` names the shorter
spelling it found.

## Running and publishing

```sh
pip install -r requirements.txt
python app.py                     # http://127.0.0.1:7860
```

`DEVICE` decides where the model runs and takes precedence over everything
else, so `DEVICE=cpu` is a local CPU test even on a machine with a GPU. With
`DEVICE` unset, the choice is CUDA when the runtime is ZeroGPU
(`SPACES_ZERO_GPU`, the marker Hugging Face sets — asking
`torch.cuda.is_available()` there answers False, because the GPU is attached
only while a `@spaces.GPU` function runs) or when torch reports a GPU, and CPU
otherwise.

To publish, from the ESPnet checkout:

```sh
hf auth login
hf upload espnet/forced-alignment egs2/owsm_ctc_v4/s2t1/demo_align . --repo-type space
```

**Not before the release named in `requirements.txt`.** A Space installs
espnet from PyPI, and `espnet2.bin.align` arrived in 202610.post2.

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
