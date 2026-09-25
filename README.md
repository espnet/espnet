<div align="center">

<img src="doc/image/espnet_logo1.png" width="440" alt="ESPnet"/>

### End-to-end speech processing toolkit

[![PyPI](https://img.shields.io/pypi/v/espnet?color=%233775A9&logo=pypi&logoColor=white)](https://pypi.org/project/espnet/)
[![Python](https://img.shields.io/pypi/pyversions/espnet.svg)](https://pypi.org/project/espnet/)
[![Downloads](https://static.pepy.tech/badge/espnet/month)](https://pepy.tech/project/espnet)
[![License](https://img.shields.io/github/license/espnet/espnet.svg?color=blue)](./LICENSE)
[![codecov](https://codecov.io/gh/espnet/espnet/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-espnet-yellow)](https://huggingface.co/espnet)
[![Discord](https://img.shields.io/discord/1174538500360650773?color=%235865F2&label=Discord&logo=discord&logoColor=white)](https://discord.gg/hrCs85gFWM)

**[Documentation](https://espnet.github.io/espnet/)** ·
**[Installation](https://espnet.github.io/espnet/installation.html)** ·
**[Recipes](egs2/)** ·
**[Model Zoo](https://huggingface.co/espnet)** ·
**[Notebooks](https://github.com/espnet/notebook)** ·
**[Discord](https://discord.gg/hrCs85gFWM)**

</div>

______________________________________________________________________

ESPnet is an end-to-end speech processing toolkit built on [PyTorch](https://pytorch.org/). It covers speech recognition, text-to-speech, speech translation, speech enhancement, speaker diarization, spoken language understanding, singing voice synthesis, speech language models, and more — with [Kaldi](http://kaldi-asr.org/)-style reproducible recipes from data preparation to evaluation, and hundreds of pretrained models on Hugging Face.

## What's new

- **[ESPnet 202610.post2](https://github.com/espnet/espnet/releases/tag/v.202610.post2)** — the command line has one name a task and two more of them: `espnet phonemize` reads the phones with POWSM, `espnet align` lines text up with the audio it was said in, and `asr` and `tts` became `transcribe` and `synthesize` (the old names still work); models published before June 2025 load again, after `init: chainer` was removed from the toolkit.

<details>
<summary>Earlier releases</summary>

- **[ESPnet 202610.post1](https://github.com/espnet/espnet/releases/tag/v.202610.post1)** — the command line grows `espnet demo` (the OWSM browser demo) and `--live` (the microphone, transcribed as you speak); one `Speech2Text` now loads either kind of OWSM checkpoint, with `best_path()` for CTC decoding without a search; `espnet/espnet:inference-cpu-latest` and `-gpu-latest` run a published model with nothing installed; three more demo Spaces (TTS, enhancement, speaker verification).
- **[ESPnet 202610](https://github.com/espnet/espnet/releases/tag/v.202610)** — one-line inference from the command line (`pip install espnet && espnet asr audio.wav`), two OWSM v4 demos as Hugging Face Spaces, a core install without the training stack (training is `espnet[train]`), batched beam search, PyTorch 2.11-2.14.
- **[ESPnet 202609](https://github.com/espnet/espnet/releases/tag/v.202609)** — ESPnet3 complete on [`egs3/librispeech_100`](egs3/librispeech_100) at ESPnet2 parity, CI rebuilt on a prebuilt image (compute per run halved), OpenBEATs pretraining, ten new recipes (ASR, TTS, SER, ST, audio SSL), Python 3.12-3.13.
- **[ESPnet 202604](https://github.com/espnet/espnet/releases/tag/v.202604)** — Docker-based CI, PyTorch 2.9.1 support, FastSpeech2 inference ~1.9x faster at batch 8, new recipes (Kinyarwanda, Emilia, kosp2e).
- **[ESPnet 202511](https://github.com/espnet/espnet/releases/tag/v.202511)** — parallel-processing primitives, refactored inference and evaluation pipeline, expanded SpeechLM support.
- **[ESPnet 202509](https://github.com/espnet/espnet/releases/tag/v.202509)** — Python 3.9-3.13, Debian 12 CI, the LID subsystem completed, multi-optimizer training (`HybridOptim` / `HybridLRS`).
- **[ESPnet 202506](https://github.com/espnet/espnet/releases/tag/v.202506)** — ESPnet3 groundwork (data organizer, trainer, model), LID training and task setup, `codec1` recipes, USES2 speech enhancement, IPAPack++ S2T recipes.
- **[ESPnet 202503](https://github.com/espnet/espnet/releases/tag/v.202503)** — PyTorch Lightning trainer support, Hugging Face front-end, scaled dot-product attention, ML-SUPERB 2024 recipe.

Full history: [Releases](https://github.com/espnet/espnet/releases).

</details>

## Install

```sh
# Install PyTorch first: https://pytorch.org/get-started/locally/
pip install espnet              # run pretrained models
pip install "espnet[train]"     # also train them (Lightning, TensorBoard, W&B, Hydra, Dask)
```

<details>
<summary>Other installation options</summary>

```sh
pip install "espnet[all]"                       # training plus every task extra (except sds)
pip install git+https://github.com/espnet/espnet  # latest master
```

- **Full setup** (recipes, DNN training, Kaldi-style tooling): see the [installation guide](https://espnet.github.io/espnet/installation.html).
- **Docker**: see [`docker/`](docker/) and the [Docker docs](https://espnet.github.io/espnet/docker.html).
- **Task-specific tools** live in [`tools/installers`](tools/installers).
- **ESPnet1 is no longer supported** — use ESPnet2 (`egs2/`) or ESPnet3 (`egs3/`). See [the ESPnet1 notice](https://espnet.github.io/espnet/espnet1_tutorial.html).

</details>

<details>
<summary>Tested environments (CI status)</summary>

|system/pytorch ver.|2.11.0|2.13.0|2.14.0|
| :---- | :---: | :---: | :---: |
|ubuntu/python3.12/pip|[![ci on ubuntu](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml?query=branch%3Amaster)|[![ci on ubuntu](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml?query=branch%3Amaster)|[![ci on ubuntu](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml?query=branch%3Amaster)|
|ubuntu/python3.13/pip|[![ci on ubuntu](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml?query=branch%3Amaster)|[![ci on ubuntu](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml?query=branch%3Amaster)|[![ci on ubuntu](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_ubuntu.yml?query=branch%3Amaster)|
|debian12/python3.12/conda|[![ci on debian12](https://github.com/espnet/espnet/actions/workflows/ci_on_debian12.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_debian12.yml?query=branch%3Amaster)|||
|windows/python3.12/pip|[![ci on windows](https://github.com/espnet/espnet/actions/workflows/ci_on_windows.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_windows.yml?query=branch%3Amaster)|||
|macos/python3.12/pip|[![ci on macos](https://github.com/espnet/espnet/actions/workflows/ci_on_macos.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_macos.yml?query=branch%3Amaster)|||
|macos/python3.12/conda|[![ci on macos](https://github.com/espnet/espnet/actions/workflows/ci_on_macos.yml/badge.svg)](https://github.com/espnet/espnet/actions/workflows/ci_on_macos.yml?query=branch%3Amaster)|||

Each badge is its workflow's aggregate status on `master`, where the full grid runs. Coverage is not uniform — some suites run on one pytorch only, and a pull request runs less than `master` does. [What each column covers](CONTRIBUTING.md#53-what-runs-on-a-pull-request).

[![pre-commit.ci](https://results.pre-commit.ci/badge/github/espnet/espnet/master.svg)](https://results.pre-commit.ci/latest/github/espnet/espnet/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Mergify](https://img.shields.io/endpoint.svg?url=https://api.mergify.com/v1/badges/espnet/espnet&style=flat)](https://mergify.com)

</details>

## Quick start

```sh
pip install espnet
```

Then pick how you want to call it. Same models, same six tasks.

**From the terminal**

```sh
espnet transcribe audio.wav                # detects the language
espnet translate audio.wav --to eng
espnet synthesize "Hello from ESPnet" -o out.wav
espnet enhance noisy.wav -o clean.wav
```

`phonemize` and `align` are there too, and `espnet models` names the default model of each. Every command but that one takes `--model <tag>` and `--device cuda`.

**From Python**

```python
from espnet2.bin.s2t_inference import Speech2Text

# OWSM-CTC v4: multilingual ASR, translation and language ID in one model
s2t = Speech2Text.from_pretrained("espnet/owsm_ctc_v4_1B")
for start, end, text in s2t.decode_long("audio.wav"):  # any length or rate
    print(text)
```

Any model from the [ESPnet organization](https://huggingface.co/espnet), cached after the first download.

**From an agent**

```sh
pip install "espnet[mcp]"
claude mcp add espnet -- espnet-mcp
```

An [MCP](https://modelcontextprotocol.io/) server offering the same six tasks as tools, so Claude, Cursor or another agent calls them itself.

<details>
<summary>Or without installing anything, in a container</summary>

```sh
docker run --rm -v "$PWD:/data" -v "$HOME/.cache/huggingface:/cache/huggingface" \
    espnet/espnet:inference-cpu-latest asr /data/audio.wav
```

The second mount keeps the downloaded model between runs. With a GPU, use `espnet/espnet:inference-gpu-latest`, `--gpus all` and `--device cuda`; on a Linux host add `--user "$(id -u):$(id -g)"`. The other images are in [`docker/`](docker/).

</details>

**Train a recipe** — every corpus follows the same interface:

```sh
cd egs2/librispeech/asr1
./run.sh                              # data → features → training → scoring
./run.sh --stage 11 --stop_stage 13   # or selected stages
```

New to ESPnet? Start with [`egs2/mini_an4/asr1`](egs2/mini_an4/asr1) — it runs end to end in minutes.

## Supported tasks

| | Task | Template | Highlights |
| :-- | :-- | :-- | :-- |
| 🗣️ | **ASR** — speech recognition | [`asr1`](egs2/TEMPLATE/asr1), [`asr2`](egs2/TEMPLATE/asr2) | Hybrid CTC/attention, Transducer, streaming, Conformer / [E-Branchformer](https://arxiv.org/abs/2210.00077), Whisper, SSL front-ends |
| 🌏 | **S2T** — multilingual multitask | [`s2t1`](egs2/TEMPLATE/s2t1) | [OWSM](https://arxiv.org/abs/2309.13876): open Whisper-style models trained on public data |
| 🔊 | **TTS** — text-to-speech | [`tts1`](egs2/TEMPLATE/tts1), [`tts2`](egs2/TEMPLATE/tts2) | Tacotron 2, FastSpeech 2, VITS, JETS, multi-speaker / multilingual |
| 🎤 | **SVS** — singing voice synthesis | [`svs1`](egs2/TEMPLATE/svs1), [`svs2`](egs2/TEMPLATE/svs2) | VISinger 1/2, Xiaoice, DiffSinger; merged from [Muskits](https://github.com/SJTMusicTeam/Muskits) |
| 🎧 | **SE/SS** — enhancement & separation | [`enh1`](egs2/TEMPLATE/enh1), [`enh_asr1`](egs2/TEMPLATE/enh_asr1) | Unified encoder–separator–decoder, TasNet / DPRNN / beamformers, ASR-integrated |
| 🌐 | **ST / MT / S2ST** — translation | [`st1`](egs2/TEMPLATE/st1), [`mt1`](egs2/TEMPLATE/mt1), [`s2st1`](egs2/TEMPLATE/s2st1) | End-to-end and cascaded speech translation, speech-to-speech translation |
| 💬 | **SLU** — language understanding | [`slu1`](egs2/TEMPLATE/slu1) | Intent + transcript multitasking, pretrained ASR/NLP encoders |
| 👤 | **SPK / LID / DIAR** — speaker & language | [`spk1`](egs2/TEMPLATE/spk1), [`lid1`](egs2/TEMPLATE/lid1), [`diar1`](egs2/TEMPLATE/diar1) | Speaker embeddings, verification, language ID, diarization |
| 🧠 | **SSL** — self-supervised learning | [`ssl1`](egs2/TEMPLATE/ssl1), [`hubert1`](egs2/TEMPLATE/hubert1) | HuBERT pretraining; [S3PRL](https://github.com/s3prl/s3prl) upstreams as front-ends |
| 🤖 | **SpeechLM** — speech language models | [`speechlm1`](egs2/TEMPLATE/speechlm1) | Unified sequence modeling across speech and text tasks |
| 📦 | **Codec** — neural audio codecs | [`codec1`](egs2/TEMPLATE/codec1) | Discrete speech tokens for downstream tasks |
| ➕ | **More** | [`uasr1`](egs2/TEMPLATE/uasr1), [`cls1`](egs2/TEMPLATE/cls1), [`asvspoof1`](egs2/TEMPLATE/asvspoof1), [`lm1`](egs2/TEMPLATE/lm1), [`sds1`](egs2/TEMPLATE/sds1) | Unsupervised ASR ([EURO](https://arxiv.org/abs/2211.17196)), audio classification, anti-spoofing, LM, spoken dialogue |

Each template ships a corpus-agnostic pipeline; see [`egs2/README.md`](egs2/README.md) for the full list of 200+ corpora recipes.

## Why ESPnet

- **Reproducible** — one `run.sh` per corpus, from download to scoring, with published results.
- **Unified** — the same recipe structure, config format, and trainer across every task above.
- **Scalable** — DDP, multi-node training, [Slurm](https://slurm.schedmd.com/)/MPI, [DeepSpeed](https://github.com/microsoft/DeepSpeed), sharded training, on-the-fly feature extraction.
- **Open** — hundreds of pretrained models and demos on [Hugging Face](https://huggingface.co/espnet), plus [W&B](https://espnet.github.io/espnet/espnet2_training_option.html#weights-biases-integration) and TensorBoard logging.

## Demos

Four ways to run a published model — a hosted [Space](https://huggingface.co/espnet), a notebook, the MCP server an agent calls, and the command line. 🟢 is there today, 🚧 is in review, ❌ is not there yet.

| Task | Space | Notebook | MCP | CLI |
| :-- | :-- | :-- | :-- | :-- |
| **ASR** — transcription | 🟢 [owsm-ctc-v4](https://huggingface.co/spaces/espnet/owsm-ctc-v4) | 🟢 [`asr_demo`](https://github.com/espnet/notebook/blob/master/Demos/asr_demo.ipynb) | 🟢 `transcribe` | 🟢 `espnet transcribe` |
| **ST** — speech translation | 🟢 [owsm-ctc-v4](https://huggingface.co/spaces/espnet/owsm-ctc-v4) | 🟢 [`st_demo`](https://github.com/espnet/notebook/blob/master/Demos/st_demo.ipynb) | 🟢 `translate` | 🟢 `espnet translate` |
| **TTS** — synthesis | 🟢 [ljspeech-vits](https://huggingface.co/spaces/espnet/ljspeech-vits) | 🟢 [`tts_demo`](https://github.com/espnet/notebook/blob/master/Demos/tts_demo.ipynb) | 🟢 `synthesize` | 🟢 `espnet synthesize` |
| **SE** — enhancement | 🟢 [universal-se](https://huggingface.co/spaces/espnet/universal-se) | 🟢 [`enh_demo`](https://github.com/espnet/notebook/blob/master/Demos/enh_demo.ipynb) | 🟢 `enhance` | 🟢 `espnet enhance` |
| **PR** — phone recognition | 🟢 [powsm-ctc](https://huggingface.co/spaces/espnet/powsm-ctc) | 🟢 [`s2t_pr_demo`](https://github.com/espnet/notebook/blob/master/Demos/s2t_pr_demo.ipynb) | 🟢 `phonemize` | 🟢 `espnet phonemize` |
| **ALIGN** — forced alignment | 🟢 [forced-alignment](https://huggingface.co/spaces/espnet/forced-alignment) | 🟢 [`s2t_align_demo`](https://github.com/espnet/notebook/blob/master/Demos/s2t_align_demo.ipynb) | 🟢 `align` | 🟢 `espnet align` |

Every Space is built from a directory in this repository, and [`doc/front_ends.md`](doc/front_ends.md) is how a seventh task gets all four columns.

`pip install "espnet[demo]"` adds `espnet demo`, the same app on localhost; `espnet transcribe --live` reads the microphone.

Beyond the six: a [Space](https://huggingface.co/spaces/espnet/speaker-verification) and a [notebook](https://github.com/espnet/notebook/blob/master/Demos/spk_demo.ipynb) for speaker verification, notebooks for [neural codecs](https://github.com/espnet/notebook/blob/master/Demos/codec_demo.ipynb) and [spoken dialogue](https://github.com/espnet/notebook/blob/master/Demos/sds_demo.ipynb), and [`egs2/TEMPLATE/sds1`](egs2/TEMPLATE/sds1) for the full spoken dialogue system, which runs locally.

<details>
<summary>Every notebook, and whether it still runs</summary>

| Demo | | Last run |
| :-- | :-- | :-- |
| Spoken dialogue — listen, think, speak | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/sds_demo.ipynb) | [![sds_demo](https://github.com/espnet/notebook/actions/workflows/sds_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/sds_demo.yml) |
| Speech recognition, in any of 151 languages | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/asr_demo.ipynb) | [![asr_demo](https://github.com/espnet/notebook/actions/workflows/asr_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/asr_demo.yml) |
| The words appearing as the audio arrives | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/asr_streaming_demo.ipynb) | [![asr_streaming_demo](https://github.com/espnet/notebook/actions/workflows/asr_streaming_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/asr_streaming_demo.yml) |
| Speech translation — the same model, a different task symbol | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/st_demo.ipynb) | [![st_demo](https://github.com/espnet/notebook/actions/workflows/st_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/st_demo.yml) |
| Text-to-speech, one voice and then 128 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/tts_demo.ipynb) | [![tts_demo](https://github.com/espnet/notebook/actions/workflows/tts_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/tts_demo.yml) |
| Speech enhancement, and what it did to the signal-to-noise | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/enh_demo.ipynb) | [![enh_demo](https://github.com/espnet/notebook/actions/workflows/enh_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/enh_demo.yml) |
| Speaker verification — two recordings, one score | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/spk_demo.ipynb) | [![spk_demo](https://github.com/espnet/notebook/actions/workflows/spk_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/spk_demo.yml) |
| Neural codecs — a waveform as a few integers a frame | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/codec_demo.ipynb) | [![codec_demo](https://github.com/espnet/notebook/actions/workflows/codec_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/codec_demo.yml) |

Each runs top to bottom on a CPU and pins the release it was checked against. The second badge is that notebook being executed cell by cell every Sunday, so a red one names the demo that broke rather than leaving you to find out by opening it. More, including the CMU course material: [espnet/notebook](https://github.com/espnet/notebook).

</details>

**Publish your own.** Every ESPnet3 recipe can wrap its trained model in a
[Gradio](https://www.gradio.app/) app and push it to Hugging Face Spaces — the UI, the Space `README.md` and `requirements.txt` are all generated from [`conf/demo.yaml`](egs3/TEMPLATE/asr/conf/demo.yaml).

<details>
<summary>The three stages</summary>

```sh
cd egs3/librispeech_100/asr
train=conf/tuning/training_e_branchformer.yaml   # the config the model was trained with
python run.py --stages pack_model  --training_config $train --publication_config conf/publication.yaml  # -> exp/.../model_pack
python run.py --stages pack_demo   --training_config $train --demo_config conf/demo.yaml                # -> demo/
python run.py --stages upload_demo --training_config $train --demo_config conf/demo.yaml                # needs `hf auth login`
```

Run the packed app locally with `python demo/app.py`.

</details>

## Learn

- [Documentation](https://espnet.github.io/espnet/) · [ESPnet2 tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- Course tutorials at CMU: [usage](https://youtu.be/YDN8cVjxSik) · [adding new models/tasks](https://youtu.be/Css3XAes7SU) ([materials](https://github.com/espnet/notebook))
- [Interspeech 2019 tutorial](https://github.com/espnet/interspeech2019-tutorial)

## Contributing

Contributions, questions, and feature requests are all welcome — open an [issue](https://github.com/espnet/espnet/issues) or a pull request. First time here? Read the [contribution guide](CONTRIBUTING.md).

<a href="https://github.com/espnet/espnet/graphs/contributors">
<img src="https://contrib.rocks/image?repo=espnet/espnet&max=100&columns=25"
alt="Contributors"/>
</a>

## Details

<details>
<summary><b>Full feature list by task</b></summary>


#### Kaldi-style complete recipe
- Support numbers of `ASR` recipes (WSJ, Switchboard, CHiME-4/5, Librispeech, TED, CSJ, AMI, HKUST, Voxforge, REVERB, Gigaspeech, etc.)
- Support numbers of `TTS` recipes in a similar manner to the ASR recipe (LJSpeech, LibriTTS, M-AILABS, etc.)
- Support numbers of `ST` recipes (Fisher-CallHome Spanish, Libri-trans, IWSLT'18, How2, Must-C, Mboshi-French, etc.)
- Support numbers of `MT` recipes (IWSLT'14, IWSLT'16, the above ST recipes etc.)
- Support numbers of `SLU` recipes (CATSLU-MAPS, FSC, Grabo, IEMOCAP, JDCINAL, SNIPS, SLURP, SWBD-DA, etc.)
- Support numbers of `SE/SS` recipes (DNS-IS2020, LibriMix, SMS-WSJ, VCTK-noisyreverb, WHAM!, WHAMR!, WSJ-2mix, etc.)
- Support voice conversion recipe (VCC2020 baseline)
- Support speaker diarization recipe (mini_librispeech, librimix)
- Support singing voice synthesis recipe (ofuton_p_utagoe_db, opencpop, m4singer, etc.)

#### ASR: Automatic Speech Recognition
- **State-of-the-art performance** in several ASR benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- **Hybrid CTC/attention** based end-to-end ASR
  - Fast/accurate training with CTC/attention multitask training
  - CTC/attention joint decoding to boost monotonic alignment decoding
  - Encoder: VGG-like CNN + BiRNN (LSTM/GRU), sub-sampling BiRNN (LSTM/GRU), Transformer, Conformer, [Branchformer](https://proceedings.mlr.press/v162/peng22a.html), or [E-Branchformer](https://arxiv.org/abs/2210.00077)
  - Decoder: RNN (LSTM/GRU), Transformer, or S4
- Attention: [Flash Attention](https://github.com/Dao-AILab/flash-attention), Dot product, location-aware attention, variants of multi-head
- Incorporate RNNLM/LSTMLM/TransformerLM/N-gram trained only with text data
- Batch GPU decoding
- Data augmentation
- **Transducer** based end-to-end ASR
  - Architecture:
    - Custom encoder supporting RNNs, Conformer, Branchformer (w/ variants), 1D Conv / TDNN.
    - Decoder w/ parameters shared across blocks supporting RNN, stateless w/ 1D Conv, [MEGA](https://arxiv.org/abs/2209.10655), and [RWKV](https://arxiv.org/abs/2305.13048).
    - Pre-encoder: VGG2L or Conv2D available.
  - Search algorithms:
    - Greedy search constrained to one emission by timestep.
    - Default beam search algorithm [[Graves, 2012]](https://arxiv.org/abs/1211.3711) without prefix search.
    - Alignment-Length Synchronous decoding [[Saon et al., 2020]](https://ieeexplore.ieee.org/abstract/document/9053040).
    - Time Synchronous Decoding [[Saon et al., 2020]](https://ieeexplore.ieee.org/abstract/document/9053040).
    - N-step Constrained beam search modified from [[Kim et al., 2020]](https://arxiv.org/abs/2002.03577).
    - modified Adaptive Expansion Search based on [[Kim et al., 2021]](https://ieeexplore.ieee.org/abstract/document/9250505) and NSC.
  - Features:
    - Unified interface for offline and streaming speech recognition.
    - Multi-task learning with various auxiliary losses:
      - Encoder: CTC, auxiliary Transducer and symmetric KL divergence.
      - Decoder: cross-entropy w/ label smoothing.
    - Transfer learning with an acoustic model and/or language model.
    - Training with FastEmit regularization method [[Yu et al., 2021]](https://arxiv.org/abs/2010.11148).
  > Please refer to the [tutorial page](https://espnet.github.io/espnet/tutorial.html#transducer) for complete documentation.
- CTC segmentation
- Non-autoregressive model based on Mask-CTC
- ASR examples for supporting endangered language documentation (see [`egs2/puebla_nahuatl`](egs2/puebla_nahuatl) and [`egs2/yoloxochitl_mixtec`](egs2/yoloxochitl_mixtec))
- Wav2Vec2.0 pre-trained model as Encoder, imported from [FairSeq](https://github.com/pytorch/fairseq/tree/master/fairseq).
- Self-supervised learning representations as features, using upstream models in [S3PRL](https://github.com/s3prl/s3prl) in frontend.
  - Set `frontend` to `s3prl`
  - Select any upstream model by setting the `frontend_conf` to the corresponding name.
- Transfer Learning :
  - easy usage and transfers from models previously trained by your group or models from [ESPnet Hugging Face repository](https://huggingface.co/espnet).
  - [Documentation](https://github.com/espnet/espnet/tree/master/egs2/mini_an4/asr1/transfer_learning.md), and an [old notebook](https://github.com/espnet/notebook/blob/master/Demos/unmaintained/asr_transfer_learning_demo.ipynb) that shows the idea. It is unmaintained and nothing runs it, so expect to fix it before it works.
- Streaming Transformer/Conformer ASR with blockwise synchronous beam search.
- Restricted Self-Attention based on [Longformer](https://arxiv.org/abs/2004.05150) as an encoder for long sequences
- OpenAI [Whisper](https://openai.com/blog/whisper/) model, robust ASR based on large-scale, weakly-supervised multitask learning

Demonstration
- Speech recognition with OWSM-CTC  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/asr_demo.ipynb)
- Hosted demo: [OWSM-CTC v4](https://huggingface.co/spaces/espnet/owsm-ctc-v4), maintained from [`egs2/owsm_ctc_v4/s2t1/demo`](egs2/owsm_ctc_v4/s2t1/demo) — recognises and identifies 151 languages, translates into 25 of them, and decodes long-form audio
- Hosted demo: [OWSM v4](https://huggingface.co/spaces/espnet/owsm-v4), maintained from [`egs2/owsm_v4/s2t1/demo`](egs2/owsm_v4/s2t1/demo) — the same four tasks with the encoder-decoder model, which also takes a text prompt
- Streaming ASR, decoded as the audio arrives [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/asr_streaming_demo.ipynb)

#### TTS: Text-to-speech
- Architecture
    - Tacotron2
    - Transformer-TTS
    - FastSpeech
    - FastSpeech2
    - Conformer FastSpeech & FastSpeech2
    - VITS
    - JETS
- Multi-speaker & multi-language extension
    - Pre-trained speaker embedding (e.g., X-vector)
    - Speaker ID embedding
    - Language ID embedding
    - Global style token (GST) embedding
    - Mix of the above embeddings
- End-to-end training
    - End-to-end text-to-wav model (e.g., VITS, JETS, etc.)
    - Joint training of text2mel and vocoder
- Various language support
    - En / Jp / Zn / De / Ru / And more...
- Integration with neural vocoders
    - Parallel WaveGAN
    - MelGAN
    - Multi-band MelGAN
    - HiFiGAN
    - StyleMelGAN
    - Mix of the above models

Demonstration
- TTS demo with ESPnet2  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/tts_demo.ipynb)
- Integrated to [Hugging Face Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/espnet/TTS)

To train the neural vocoder, please check the following repositories:
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

#### SE: Speech enhancement (and separation)

- Single-speaker speech enhancement
- Multi-speaker speech separation
- Unified encoder-separator-decoder structure for time-domain and frequency-domain models
  - Encoder/Decoder: STFT/iSTFT, Convolution/Transposed-Convolution
  - Separators: BLSTM, Transformer, Conformer, [TasNet](https://arxiv.org/abs/1809.07454), [DPRNN](https://arxiv.org/abs/1910.06379), [SkiM](https://arxiv.org/abs/2201.10800), [SVoice](https://arxiv.org/abs/2011.02329), [DC-CRN](https://web.cse.ohio-state.edu/~wang.77/papers/TZW.taslp21.pdf), [DCCRN](https://arxiv.org/abs/2008.00264), [Deep Clustering](https://ieeexplore.ieee.org/document/7471631), [Deep Attractor Network](https://pubmed.ncbi.nlm.nih.gov/29430212/), [FaSNet](https://arxiv.org/abs/1909.13387), [iFaSNet](https://arxiv.org/abs/1910.14104), Neural Beamformers, etc.
- Flexible ASR integration: working as an individual task or as the ASR frontend
- Easy to import pre-trained models from [Asteroid](https://github.com/asteroid-team/asteroid)
  - Both the pre-trained models from Asteroid and the specific configuration are supported.

Demonstration
- Interactive SE demo with ESPnet2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fjRJCh96SoYLZPRxsjF9VDv4Q2VoIckI?usp=sharing)
- Streaming SE demo with ESPnet2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17vd1V78eJpp3PHBnbFE5aVY5uMxQFL6o?usp=sharing)

#### ST: Speech Translation & MT: Machine Translation
- **State-of-the-art performance** in several ST benchmarks (comparable/superior to cascaded ASR and MT)
- Transformer-based end-to-end ST (new!)
- Transformer-based end-to-end MT (new!)

#### VC: Voice conversion
- Transformer and Tacotron2-based parallel VC using Mel spectrogram
- End-to-end VC based on cascaded ASR+TTS (Baseline system for Voice Conversion Challenge 2020!)

#### SLU: Spoken Language Understanding
- Architecture
    - Transformer-based Encoder
    - Conformer-based Encoder
    - [Branchformer](https://proceedings.mlr.press/v162/peng22a.html) based Encoder
    - [E-Branchformer](https://arxiv.org/abs/2210.00077) based Encoder
    - RNN based Decoder
    - Transformer-based Decoder
- Support Multitasking with ASR
    - Predict both intent and ASR transcript
- Support Multitasking with NLU
    - Deliberation encoder based 2 pass model
- Support using pre-trained ASR models
    - Hubert
    - Wav2vec2
    - VQ-APC
    - TERA and more ...
- Support using pre-trained NLP models
    - BERT
    - MPNet And more...
- Various language support
    - En / Jp / Zn / Nl / And more...
- Supports using context from previous utterances
- Supports using other tasks like SE in a pipeline manner
- Supports Two Pass SLU that combines audio and ASR transcript
Demonstration
- Performing noisy spoken language understanding using a speech enhancement model followed by a spoken language understanding model.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14nCrJ05vJcQX0cJuXjbMVFWUHJ3Wfb6N?usp=sharing)
- Performing two-pass spoken language understanding where the second pass model attends to both acoustic and semantic information.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p2cbGIPpIIcynuDl4ZVHDpmNPl8Nh_ci?usp=sharing)


#### SUM: Speech Summarization
- End to End Speech Summarization Recipe for Instructional Videos using Restricted Self-Attention [[Sharma et al., 2022]](https://arxiv.org/abs/2110.06263)

#### SVS: Singing Voice Synthesis
- Framework merge from [Muskits](https://github.com/SJTMusicTeam/Muskits)
- Architecture
  - RNN-based non-autoregressive model
  - Xiaoice
  - Tacotron-singing
  - DiffSinger (in progress)
  - VISinger
  - VISinger 2 (its variations with different vocoders-architecture)
- Support multi-speaker & multilingual singing synthesis
  - Speaker ID embedding
  - Language ID embedding
- Various language support
  - Jp / En / Kr / Zh
- Tight integration with neural vocoders (the same as TTS)

#### SSL: Self-supervised Learning
- Support HuBERT Pre-training:
  * Example recipe: [egs2/LibriSpeech/ssl1](egs2/LibriSpeech/ssl1)

#### UASR: Unsupervised ASR (EURO: ESPnet Unsupervised Recognition - Open-source)
- Architecture
  - wav2vec-U (with different self-supervised models)
  - wav2vec-U 2.0 (in progress)
- Support PrefixBeamSearch and K2-based WFST decoding

#### S2T: Speech-to-text with Whisper-style multilingual multitask models
- Reproduces Whisper-style training from scratch using public data: [OWSM](https://arxiv.org/abs/2309.13876)
- Supports multiple tasks in a single model
  - Multilingual speech recognition
  - Any-to-any speech translation
  - Language identification
  - Utterance-level timestamp prediction (segmentation)

#### DNN Framework
- Flexible network architecture on PyTorch
- Flexible front-end processing thanks to [omniio](https://github.com/wavlab-speech/omniio) and HDF5 support
- Tensorboard-based monitoring
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)-based large-scale training

#### ESPnet2
See [ESPnet2](https://espnet.github.io/espnet/espnet2_tutorial.html).

- Independent from Kaldi/Chainer, unlike ESPnet1
- On-the-fly feature extraction and text processing when training
- Supporting DistributedDataParallel and DaraParallel both
- Supporting multiple nodes training and integrated with [Slurm](https://slurm.schedmd.com/) or MPI
- Supporting Sharded Training provided by [fairscale](https://github.com/facebookresearch/fairscale)
- A template recipe that can be applied to all corpora
- Possible to train any size of corpus without CPU memory error
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)
- Integrated with [wandb](https://espnet.github.io/espnet/espnet2_training_option.html#weights-biases-integration)

</details>

<details>
<summary><b>Benchmark results and command-line demos</b></summary>

#### ASR results

<details><summary>expand</summary><div>


We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

| Task                                                              |     CER (%)     |     WER (%)     |                                                                              Pre-trained model                                                                               |
| ----------------------------------------------------------------- | :-------------: | :-------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Aishell dev/test                                      |     4.1/4.4     |       N/A       |                [link](https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1#branchformer-initial)                                                                  |
| CSJ eval1/eval2/eval3                                 |   4.5/3.3/3.6   |       N/A       |                                        [link](https://github.com/espnet/espnet/tree/master/egs2/csj/asr1#initial-conformer-results)                                         |
| GigaSpeech dev/test                                   |       N/A       |    10.6/10.5    |                                          [link](https://github.com/espnet/espnet/tree/master/egs2/gigaspeech/asr1#e-branchformer)                                           |
| HKUST dev                                             |      21.2       |       N/A       |                                    [link](https://github.com/espnet/espnet/tree/master/egs2/hkust/asr1#transformer-asr--transformer-lm)                                     |
| Librispeech dev_clean/dev_other/test_clean/test_other | 0.6/1.5/0.6/1.4 | 1.7/3.4/1.8/3.6 |    [link](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#self-supervised-learning-features-hubert_large_ll60k-conformer-utt_mvn-with-transformer-lm)    |
| Switchboard (eval2000) callhm/swbd                    |       N/A       |    13.4/7.3     |                                             [link](https://github.com/espnet/espnet/tree/master/egs2/swbd/asr1#e-branchformer)                                              |
| TEDLIUM2 dev/test                                     |       N/A       |     7.3/7.1     |                 [link](https://github.com/espnet/espnet/blob/master/egs2/tedlium2/asr1/README.md#e-branchformer-12-encoder-layers)                                          |
| WSJ dev93/eval92                                      |     1.1/0.8     |     2.8/1.8     |       [link](https://github.com/espnet/espnet/tree/master/egs2/wsj/asr1#self-supervised-learning-features-wav2vec2_large_ll60k-conformer-utt_mvn-with-transformer-lm)       |


If you want to check the results of the other recipes, please check `egs2/<name_of_recipe>/asr1/README.md`.

</div></details>


#### SE results
<details><summary>expand</summary><div>

We list results from three different models on WSJ0-2mix, which is one the most widely used benchmark dataset for speech separation.

| Model                                             | STOI | SAR   | SDR   | SIR   |
| ------------------------------------------------- | ---- | ----- | ----- | ----- |
| [TF Masking](https://zenodo.org/record/4498554)   | 0.89 | 11.40 | 10.24 | 18.04 |
| [Conv-Tasnet](https://zenodo.org/record/4498562)  | 0.95 | 16.62 | 15.94 | 25.90 |
| [DPRNN-Tasnet](https://zenodo.org/record/4688000) | 0.96 | 18.82 | 18.29 | 28.92 |

</div></details>

#### MT results

<details><summary>expand</summary><div>

| Task                                              | BLEU  |                                                                        Pre-trained model                                                                         |
| ------------------------------------------------- | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| IWSLT'14 test2014 (De->En)                        | 32.2  | [link](https://github.com/espnet/espnet/blob/master/egs2/iwslt14/mt1/README.md)  |

</div></details>

#### TTS results

<details><summary>expand</summary><div>

You can listen to the generated samples in the following URL.
- [ESPnet2 TTS generated samples](https://drive.google.com/drive/folders/1H3fnlBbWMEkQUfrHqosKN_ZX_WjO29ma?usp=sharing)

> Note that in the generation, we use Griffin-Lim (`wav/`) and [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) (`wav_pwg/`).

You can download pre-trained models via `espnet_model_zoo`.
- [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo)
- [Pre-trained model list](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv)

You can download pre-trained vocoders via `kan-bayashi/ParallelWaveGAN`.
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [Pre-trained vocoder list](https://github.com/kan-bayashi/ParallelWaveGAN#results)

</div></details>

#### TTS demo

<details><summary>expand</summary><div>

You can try the real-time demo in Google Colab.
Please access the notebook from the following button and enjoy the real-time synthesis!

- TTS demo with ESPnet2  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/Demos/tts_demo.ipynb)

English, Japanese, and Mandarin models are available in the demo.

</div></details>

#### VC results

<details><summary>expand</summary><div>

- Transformer and Tacotron2-based VC

You can listen to some samples on the [demo webpage](https://unilight.github.io/Publication-Demos/publications/transformer-vc/).

- Cascade ASR+TTS as one of the baseline systems of VCC2020

The [Voice Conversion Challenge 2020](http://www.vc-challenge.org/) (VCC2020) adopts ESPnet to build an end-to-end based baseline system.
In VCC2020, the objective is intra/cross-lingual nonparallel VC.
You can download converted samples of the cascade ASR+TTS baseline system [here](https://drive.google.com/drive/folders/1oeZo83GrOgtqxGwF7KagzIrfjr8X59Ue?usp=sharing).

</div></details>

#### SLU results

<details><summary>expand</summary><div>


We list the performance on various SLU tasks and datasets using the metric reported in the original dataset paper

| Task                                                              | Dataset                                                              |    Metric     |     Result     |                                                                              Pre-trained Model                                         |
| ----------------------------------------------------------------- | :-------------: | :-------------: | :-------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Intent Classification                                                 |     SLURP     |       Acc       |       86.3       |                [link](https://github.com/espnet/espnet/tree/master/egs2/slurp/asr1/README.md)                |
| Intent Classification                                                   |     FSC     |       Acc       |       99.6       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc/asr1/README.md)                |
| Intent Classification                                                  |     FSC Unseen Speaker Set     |       Acc       |       98.6       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_unseen/asr1/README.md)                |
| Intent Classification                                                   |     FSC Unseen Utterance Set     |       Acc       |       86.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_unseen/asr1/README.md)                |
| Intent Classification                                                   |     FSC Challenge Speaker Set     |       Acc       |       97.5       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_challenge/asr1/README.md)                |
| Intent Classification                                                   |     FSC Challenge Utterance Set     |       Acc       |       78.5       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_challenge/asr1/README.md)                |
| Intent Classification                                                   |     SNIPS     |       F1       |       91.7       |                [link](https://github.com/espnet/espnet/tree/master/egs2/snips/asr1/README.md)                |
| Intent Classification                                                   |     Grabo (Nl)   |       Acc       |       97.2       |                [link](https://github.com/espnet/espnet/tree/master/egs2/grabo/asr1/README.md)                |
| Intent Classification                                                   |     CAT SLU MAP (Zn)     |       Acc       |       78.9       |                [link](https://github.com/espnet/espnet/tree/master/egs2/catslu/asr1/README.md)                |
| Intent Classification                                                  |     Google Speech Commands    |       Acc       |       98.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/speechcommands/asr1/README.md)                |
| Slot Filling                                                  |     SLURP     |       SLU-F1       |       71.9       |                [link](https://github.com/espnet/espnet/tree/master/egs2/slurp_entity/asr1/README.md)                |
| Dialogue  Act Classification                                                 |     Switchboard     |       Acc       |       67.5       |                [link](https://github.com/espnet/espnet/tree/master/egs2/swbd_da/asr1/README.md)                |
| Dialogue  Act Classification                                                 |     Jdcinal (Jp)    |       Acc       |       67.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/jdcinal/asr1/README.md)                |
| Emotion Recognition                                                  |     IEMOCAP     |       Acc       |       69.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/iemocap/asr1/README.md)                |
| Emotion Recognition                                                  |     swbd_sentiment     |       Macro F1       |       61.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/swbd_sentiment/asr1/README.md)                |
| Emotion Recognition                                                  |     slue_voxceleb     |       Macro F1       |       44.0       |                [link](https://github.com/espnet/espnet/tree/master/egs2/slue-voxceleb/asr1/README.md)                |


If you want to check the results of the other recipes, please check `egs2/<name_of_recipe>/asr1/RESULTS.md`.


</div></details>

#### CTC Segmentation demo

<details><summary>expand</summary><div>

[CTC segmentation](https://arxiv.org/abs/2007.09127) determines utterance segments within audio files.
Aligned utterance segments constitute the labels of speech datasets.

As a demo, we align the start and end of utterances within the audio file `ctc_align_test.wav`.
This can be done either directly from the Python command line or using the script `espnet2/bin/asr_align.py`.
Import `CTCSegmentation` from `espnet2.bin.asr_align` for an ASR model and from `espnet2.bin.s2t_align` for an OWSM-CTC one:
those are the two entry points, and both run the algorithm in `espnet2/bin/ctc_segment.py`, which is not instantiated on its own.
For a forced alignment over any CTC head instead, and for what `espnet align` runs, see `espnet2/bin/align.py`.

From the Python command line interface:

```python
# load a model with character tokens
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader(cachedir="./modelcache")
wsjmodel = d.download_and_unpack("kamo-naoyuki/wsj")
# load the example file included in the ESPnet repository
import soundfile
speech, rate = soundfile.read("./test_utils/ctc_align_test.wav")
# CTC segmentation
from espnet2.bin.asr_align import CTCSegmentation
aligner = CTCSegmentation( **wsjmodel , fs=rate )
text = """
utt1 THE SALE OF THE HOTELS
utt2 IS PART OF HOLIDAY'S STRATEGY
utt3 TO SELL OFF ASSETS
utt4 AND CONCENTRATE ON PROPERTY MANAGEMENT
"""
segments = aligner(speech, text)
print(segments)
# utt1 utt 0.26 1.73 -0.0154 THE SALE OF THE HOTELS
# utt2 utt 1.73 3.19 -0.7674 IS PART OF HOLIDAY'S STRATEGY
# utt3 utt 3.19 4.20 -0.7433 TO SELL OFF ASSETS
# utt4 utt 4.20 6.10 -0.4899 AND CONCENTRATE ON PROPERTY MANAGEMENT
```

Aligning also works with fragments of the text.
For this, set the `gratis_blank` option that allows skipping unrelated audio sections without penalty.
It's also possible to omit the utterance names at the beginning of each line by setting `kaldi_style_text` to False.

```python
aligner.set_config( gratis_blank=True, kaldi_style_text=False )
text = ["SALE OF THE HOTELS", "PROPERTY MANAGEMENT"]
segments = aligner(speech, text)
print(segments)
# utt_0000 utt 0.37 1.72 -2.0651 SALE OF THE HOTELS
# utt_0001 utt 4.70 6.10 -5.0566 PROPERTY MANAGEMENT
```

The script `espnet2/bin/asr_align.py` uses a similar interface. To align utterances:

```sh
# ASR model and config files from pre-trained model (e.g., from cachedir):
asr_config=<path-to-model>/config.yaml
asr_model=<path-to-model>/valid.*best.pth
# prepare the text file
wav="test_utils/ctc_align_test.wav"
text="test_utils/ctc_align_text.txt"
cat << EOF > ${text}
utt1 THE SALE OF THE HOTELS
utt2 IS PART OF HOLIDAY'S STRATEGY
utt3 TO SELL OFF ASSETS
utt4 AND CONCENTRATE
utt5 ON PROPERTY MANAGEMENT
EOF
# obtain alignments:
python espnet2/bin/asr_align.py --asr_train_config ${asr_config} --asr_model_file ${asr_model} --audio ${wav} --text ${text}
# utt1 ctc_align_test 0.26 1.73 -0.0154 THE SALE OF THE HOTELS
# utt2 ctc_align_test 1.73 3.19 -0.7674 IS PART OF HOLIDAY'S STRATEGY
# utt3 ctc_align_test 3.19 4.20 -0.7433 TO SELL OFF ASSETS
# utt4 ctc_align_test 4.20 4.97 -0.6017 AND CONCENTRATE
# utt5 ctc_align_test 4.97 6.10 -0.3477 ON PROPERTY MANAGEMENT
```

The output of the script can be redirected to a `segments` file by adding the argument `--output segments`.
Each line contains the file/utterance name, utterance start and end times in seconds, and a confidence score; optionally also the utterance text.
The confidence score is a probability in log space that indicates how well the utterance was aligned. If needed, remove bad utterances:

```sh
min_confidence_score=-7
# here, we assume that the output was written to the file `segments`
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' segments
```

See the module documentation for more information.
It is recommended to use models with RNN-based encoders (such as BLSTMP) for aligning large audio files;
rather than using Transformer models that have a high memory consumption on longer audio data.
The sample rate of the audio must be consistent with that of the data used in training; adjust with `sox` if needed.

Also, we can use this tool to provide token-level segmentation information if we prepare a list of tokens instead of that of utterances in the `text` file. See the discussion in https://github.com/espnet/espnet/issues/4278#issuecomment-1100756463.

</div></details>

</details>

## Citation

If you use ESPnet in your research, please cite the main paper:

```bibtex
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
```

<details>
<summary>Task-specific papers (TTS, ST, SE, SLU, SVS, UASR, S2T, SUM, SPK)</summary>

```bibtex
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
@article{hayashi2021espnet2,
  title={{ESP}net2-{TTS}: Extending the edge of {TTS} research},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Yoshimura, Takenori and Wu, Peter and Shi, Jiatong and Saeki, Takaaki and Ju, Yooncheol and Yasuda, Yusuke and Takamichi, Shinnosuke and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2110.07840},
  year={2021}
}
@inproceedings{li2020espnet,
  title={{ESPnet-SE}: End-to-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph Boeddeker and Zhuo Chen and Shinji Watanabe},
  booktitle={Proceedings of IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE},
}
@inproceedings{arora2021espnet,
  title={{ESPnet-SLU}: Advancing Spoken Language Understanding through ESPnet},
  author={Arora, Siddhant and Dalmia, Siddharth and Denisov, Pavel and Chang, Xuankai and Ueda, Yushi and Peng, Yifan and Zhang, Yuekai and Kumar, Sujay and Ganesan, Karthik and Yan, Brian and others},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7167--7171},
  year={2022},
  organization={IEEE}
}
@inproceedings{shi2022muskits,
  author={Shi, Jiatong and Guo, Shuai and Qian, Tao and Huo, Nan and Hayashi, Tomoki and Wu, Yuning and Xu, Frank and Chang, Xuankai and Li, Huazhe and Wu, Peter and Watanabe, Shinji and Jin, Qin},
  title={{Muskits}: an End-to-End Music Processing Toolkit for Singing Voice Synthesis},
  year={2022},
  booktitle={Proceedings of Interspeech},
  pages={4277-4281},
  url={https://www.isca-speech.org/archive/pdfs/interspeech_2022/shi22d_interspeech.pdf}
}
@inproceedings{lu22c_interspeech,
  author={Yen-Ju Lu and Xuankai Chang and Chenda Li and Wangyou Zhang and Samuele Cornell and Zhaoheng Ni and Yoshiki Masuyama and Brian Yan and Robin Scheibler and Zhong-Qiu Wang and Yu Tsao and Yanmin Qian and Shinji Watanabe},
  title={{ESPnet-SE++: Speech Enhancement for Robust Speech Recognition, Translation, and Understanding}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={5458--5462},
}
@inproceedings{gao2023euro,
  title={{EURO: ESP}net unsupervised {ASR} open-source toolkit},
  author={Gao, Dongji and Shi, Jiatong and Chuang, Shun-Po and Garcia, Leibny Paola and Lee, Hung-yi and Watanabe, Shinji and Khudanpur, Sanjeev},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
@inproceedings{peng2023reproducing,
  title={Reproducing {W}hisper-style training using an open-source toolkit and publicly available data},
  author={Peng, Yifan and Tian, Jinchuan and Yan, Brian and Berrebbi, Dan and Chang, Xuankai and Li, Xinjian and Shi, Jiatong and Arora, Siddhant and Chen, William and Sharma, Roshan and others},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
@inproceedings{sharma2023espnet,
  title={ESPnet-{SUMM}: Introducing a novel large dataset, toolkit, and a cross-corpora evaluation of speech summarization systems},
  author={Sharma, Roshan and Chen, William and Kano, Takatomo and Sharma, Ruchira and Arora, Siddhant and Watanabe, Shinji and Ogawa, Atsunori and Delcroix, Marc and Singh, Rita and Raj, Bhiksha},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
@article{jung2024espnet,
  title={{ESPnet-SPK}: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models},
  author={Jung, Jee-weon and Zhang, Wangyou and Shi, Jiatong and Aldeneh, Zakaria and Higuchi, Takuya and Theobald, Barry-John and Abdelaziz, Ahmed Hussen and Watanabe, Shinji},
  journal={Proc. Interspeech 2024},
  year={2024}
}
@inproceedings{yan-etal-2023-espnet,
    title = "{ESP}net-{ST}-v2: Multipurpose Spoken Language Translation Toolkit",
    author = "Yan, Brian  and
      Shi, Jiatong  and
      Tang, Yun  and
      Inaguma, Hirofumi  and
      Peng, Yifan  and
      Dalmia, Siddharth  and
      Pol{\'a}k, Peter  and
      Fernandes, Patrick  and
      Berrebbi, Dan  and
      Hayashi, Tomoki  and
      Zhang, Xiaohui  and
      Ni, Zhaoheng  and
      Hira, Moto  and
      Maiti, Soumi  and
      Pino, Juan  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    pages = "400--411",
}
@inproceedings{someki2022espnet,
  title={{ESPnet-ONNX}: Bridging a gap between research and production},
  author={Someki, Masao and Higuchi, Yosuke and Hayashi, Tomoki and Watanabe, Shinji},
  booktitle={2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  year={2022},
  organization={IEEE}
}
@inproceedings{someki2024espnet,
  title={{ESPnet-EZ}: Python-only {ESPnet} for easy fine-tuning and integration},
  author={Someki, Masao and Choi, Kwanghee and Arora, Siddhant and Chen, William and Cornell, Samuele and Han, Jionghao and Peng, Yifan and Shi, Jiatong and Srivastav, Vaibhav and Watanabe, Shinji},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
  year={2024},
  organization={IEEE}
}
@inproceedings{shi2024espnet,
  title={{ESPnet-Codec}: Comprehensive Training and Evaluation of Neural Codecs for Audio, Music, and Speech},
  author={Shi, Jiatong and Tian, Jinchuan and Wu, Yihan and Jung, Jee-weon and Yip, Jia Qi and Masuyama, Yoshiki and Chen, William and Wu, Yuning and Tang, Yuxun and Baali, Massa and Alharhi, Dareen and Zhang, Dong and Deng, Ruifan and Srivastava, Tejes and Wu, Haibin and Liu, Alexander H. and Raj, Bhiksha and Jin, Qin and Song, Ruihua and Watanabe, Shinji},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
  pages={562--569},
  year={2024},
  organization={IEEE}
}
@inproceedings{tian-etal-2025-espnet,
  title={{ESP}net-{S}peech{LM}: An Open Speech Language Model Toolkit},
  author={Tian, Jinchuan and Shi, Jiatong and Chen, William and Arora, Siddhant and Masuyama, Yoshiki and Maekaku, Takashi and Wu, Yihan and Peng, Junyi and Bharadwaj, Shikhar and Zhao, Yiwen and Cornell, Samuele and Peng, Yifan and Yue, Xiang and Yang, Chao-Han Huck and Neubig, Graham and Watanabe, Shinji},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (System Demonstrations)},
  pages={116--124},
  year={2025},
  publisher={Association for Computational Linguistics}
}
@inproceedings{arora-etal-2025-espnet,
  title={{ESP}net-{SDS}: Unified Toolkit and Demo for Spoken Dialogue Systems},
  author={Arora, Siddhant and Peng, Yifan and Shi, Jiatong and Tian, Jinchuan and Chen, William and Bharadwaj, Shikhar and Futami, Hayato and Kashiwagi, Yosuke and Tsunoo, Emiru and Shimizu, Shuichiro and Srivastav, Vaibhav and Watanabe, Shinji},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (System Demonstrations)},
  pages={248--259},
  year={2025},
  publisher={Association for Computational Linguistics}
}
@inproceedings{bharadwaj2025openbeats,
  title={{OpenBEATs}: A Fully Open-Source General-Purpose Audio Encoder},
  author={Bharadwaj, Shikhar and Cornell, Samuele and Choi, Kwanghee and Fukayama, Satoru and Shim, Hye-jin and Deshmukh, Soham and Watanabe, Shinji},
  booktitle={2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2025},
  organization={IEEE}
}
@inproceedings{someki2026espnet3,
  title={{ESPnet3}: Infrastructure for Scalable Speech and Audio Research in the Foundation Model Era},
  author={Someki, Masao and Polok, Alexander and Carvalho, Carlos and Lin, Chyi-Jiunn and Yang, Da-Hee and Shi, Jiatong and Tian, Jinchuan and Yalta Soplin, Nelson Enrique and Cornell, Samuele and Arora, Siddhant and Teixeira, Francisco and Wang, Wei and Chen, William and Abad, Alberto and Li, Chenda and Watanabe, Shinji and Zhang, Wangyou},
  booktitle={Interspeech 2026},
  year={2026}
}
```

</details>

______________________________________________________________________

<div align="center">
Released under the <a href="./LICENSE">Apache 2.0 License</a>.
</div>
