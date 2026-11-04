---
title: POWSM-CTC
emoji: 🐁
colorFrom: purple
colorTo: pink
sdk: gradio
python_version: "3.12"
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: What you said in phones, IPA, from a phonetic model
tags:
  - espnet
  - owsm
  - powsm
  - phone-recognition
  - automatic-speech-recognition
models:
  - espnet/powsm_ctc
---

# POWSM-CTC

The source of [`espnet/powsm-ctc`](https://huggingface.co/spaces/espnet/powsm-ctc),
the demo of a phonetic model: speech in, the phones that were said out, in
IPA. [POWSM](https://arxiv.org/abs/2510.24992) is the first phonetic
foundation model; POWSM-CTC is the encoder-only variant, trained on
[IPAPack++](https://huggingface.co/anyspeech) by the recipe in the directory
above this one and released with
[PRiSM](https://arxiv.org/abs/2601.14046), a benchmark for phone recognisers.

It is the model `espnet phonemize` uses by default, and the one the MCP
server's `phonemize` tool loads.

## What is here, and what is not

This app has no interface of its own. The page is
[`espnet2.bin.demo`](../../../../espnet2/bin/demo.py), the module `espnet
demo` serves, and it reads what to offer off the checkpoint:

| | read from the model |
|---|---|
| the language menu | the symbols before `<asr>` in its token list |
| the tasks | the four symbols this checkpoint has: `<asr>`, `<pr>`, `<g2p>`, `<p2g>` |
| the written-input box | shown for `<g2p>` and `<p2g>`, which take something besides the audio |
| the decoding window | 20 s for POWSM, where OWSM is 30 |
| "detect the language" | this checkpoint spells it `<unk>` rather than `<nolang>` |

POWSM does four things with one recording, and the page offers all four:

| task | you give | it answers |
|---|---|---|
| Transcribe | the audio | the words |
| Recognise phones | the audio | the phones, in IPA |
| Phones for text you give (G2P) | the audio and the words that were said | the phones |
| Text for phones you give (P2G) | the audio and the phones | the words |

The last two are the model's own `<g2p>` and `<p2g>`, trained with the
written half as the prompt (`text.prev` in the recipe). They read one window,
since the prompt belongs to one utterance.

**They are the weaker half of this checkpoint, and the page says so.** POWSM's
author put it plainly on [#6792](https://github.com/espnet/espnet/pull/6792):
`<g2p>` and `<p2g>` are encoder-decoder work, and an encoder-CTC model is less
stable at them - [POWSM](https://huggingface.co/espnet/powsm) is what to reach
for if you need them, at four to twelve times the runtime. For a user the two
are also close to the tasks above: `<g2p>` is `<pr>` with the words typed in
first, and `<p2g>` is `<asr>` with the phones typed in first. They are here
because the model was trained with them and the page reads its token list, not
because this checkpoint is good at them.

So `python app.py` and `espnet demo --model espnet/powsm_ctc` are the same
page, and a fix to either is a fix to both. The two OWSM demos beside this
one still carry their own copies of that code, written before the module
existed; `test/espnet2/bin/test_demo_apps.py` is what keeps those from
drifting, and this app needs none of it.

The page says, where a reader will see it, that this is a phonetic model:
its transcription is weaker than a model trained for text. Measured on
`test_utils/ctc_align_test.wav`, on a laptop CPU, 6 s of audio: **Recognise
phones** gives `ð ə s e ɪ l ɔ v ð ə h o t ɛ l s …` in 6.2 s, and **Transcribe**
gives "desel of the hotels is part of holiday strategy …" in 5.7 s. The
language menu is the checkpoint's own and so is its language identification,
which on that file names Kurmanji; the phones are right regardless, since
they are what it was trained to produce.

## Which POWSM

There are two, and `POWSM_MODEL_TAG` chooses between them without touching a
file - `espnet/powsm_ctc` here, `espnet/powsm` for the encoder-decoder.

Measured on `test_utils/ctc_align_test.wav`, on a laptop CPU, through
`espnet2.bin.s2t_inference`:

| | `powsm_ctc` | `powsm` |
|---|---|---|
| `<pr>` | 5.3 s | 21.8 s, and a different reading of the same phones - `ð ə s e ɪ l ʌ v` where the CTC wrote `d ə s e ɪ l ɔ v` |
| `<asr>` | 7.0 s | 53.4 s, and the text comes back as `T ⁇ E  ⁇ A ⁇ E OF …` |
| `<g2p>` | 5.9 s | 65.7 s |
| `<p2g>` | 7.6 s | 89.4 s, same `⁇` |

Both write the same phone set - a diphthong is two symbols in either, by
design - so the difference above is which phones each chose on this file,
not what either can say.

The hosted Space is the CTC one: 89 s against a 120 s ZeroGPU slice leaves no
room, and the encoder-decoder's two text tasks are the ones its text
normalisation is hurting today. Asked with text the audio does not say, it
was the CTC model whose phones followed the prompt and the encoder-decoder's
that did not change - so on this file the prompted tasks are the CTC one's
strength rather than its weakness. One file, no metric; see
[#6792](https://github.com/espnet/espnet/pull/6792) for the numbers and the
caveats.

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
hf upload espnet/powsm-ctc egs2/powsm_ctc/s2t1/demo . --repo-type space
```

**Not before the release named in `requirements.txt`.** A Space installs
espnet from PyPI, and this app imports `espnet2.bin.demo`, whose phone page
and `wrap` argument arrived in 202610.post2. Uploading before that release is
a Space that builds and then fails to start.

## Citation

Copied from the recipe above, which is where they are maintained:

```bibtex
@article{powsm,
      title={POWSM: A Phonetic Open Whisper-Style Speech Foundation Model},
      author={Chin-Jou Li and Kalvin Chang and Shikhar Bharadwaj and Eunjung Yeo and Kwanghee Choi and Jian Zhu and David Mortensen and Shinji Watanabe},
      year={2025},
      eprint={2510.24992},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.24992},
}

@article{prism,
      title={PRiSM: Benchmarking Phone Realization in Speech Models},
      author={Shikhar Bharadwaj and Chin-Jou Li and Yoonjae Kim and Kwanghee Choi and Eunjung Yeo and Ryan Soh-Eun Shim and Hanyu Zhou and Brendon Boldt and Karen Rosero Jacome and Kalvin Chang and Darsh Agrawal and Keer Xu and Chao-Han Huck Yang and Jian Zhu and Shinji Watanabe and David R. Mortensen},
      year={2026},
      eprint={2601.14046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.14046},
}
```
