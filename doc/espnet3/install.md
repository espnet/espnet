---
title: ESPnet3 Installation
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Installation

ESPnet3 uses the same Python package name as ESPnet2: `espnet`.

## Quick install (pip)

```bash
pip install espnet
```

System extras (ASR/TTS/ST/ENH, etc.):

```bash
pip install "espnet[asr]"
pip install "espnet[tts]"
pip install "espnet[st]"
pip install "espnet[enh]"
```

Supported extras (from `pyproject.toml`):

| Extra | Packages | Description |
| --- | --- | --- |
| `asr` | `ctc-segmentation`, `editdistance`, `opt_einsum`, `jiwer` | ASR alignment and scoring (e.g., WER). |
| `tts` | `pyworld`, `pypinyin`, `espnet_tts_frontend`, `g2p_en`, `jamo`, `jaconv` | TTS frontends, G2P, and language processing. |
| `enh` | `ci_sdr`, `fast-bss-eval` | Speech enhancement metrics. |
| `asr2` | `editdistance` | ESPnet2-style ASR extras. |
| `s2st` | `editdistance`, `s3prl` | Speech-to-speech translation + SSL features. |
| `st` | `editdistance` | Speech translation scoring. |
| `s2t` | `editdistance` | Speech-to-text translation scoring. |
| `spk` | `asteroid_filterbanks` | Speaker tasks. |
| `dev` | `black`, `flake8`, `pytest`, `pytest-cov`, `isort` | Developer tooling (format/lint/test). |
| `test` | `pytest`, `pytest-timeouts`, `pytest-pythonpath`, `pytest-cov`, `hacking`, `mock`, `pycodestyle`, `jsondiff`, `flake8`, `flake8-docstrings`, `black`, `isort`, `h5py` | Test stack used by CI. |
| `doc` | `sphinx`, `sphinx-rtd-theme`, `myst-parser`, `sphinx-argparse`, `sphinx-markdown-builder`, `sphinx-markdown-tables` | Documentation build tools. |
| `all` | `espnet[asr]`, `espnet[tts]`, `espnet[enh]`, `espnet[spk]`, `fairscale`, `transformers`, `evaluate` | Convenience meta extra. |

## Using uv

[uv](https://docs.astral.sh/uv/) is fast and works well for reproducible Python environments.
It makes it easy to pin Python versions and manage isolated virtualenvs without
system-wide installs.

```bash
uv venv .venv
. .venv/bin/activate
uv pip install espnet
```

For extras:

```bash
uv pip install "espnet[asr]"
```

## Using pixi

[Pixi](https://pixi.prefix.dev/latest/#highlights) can manage Python and system dependencies together.
This lets you install packages that previously required conda-forge entirely in
user space, without system-level package managers.

```bash
pixi init
pixi add python=3.10 pip
pixi run pip install espnet
```

## Install from source (recommended for development)

```bash
git clone https://github.com/espnet/espnet.git
cd espnet/tools
. setup_uv.sh
```

Then install the editable package with extras as needed:

```bash
cd ..
uv pip install -e ".[asr]"
```

## Recipe tool installers (optional)

Some recipes rely on external tools (e.g., `sph2pipe`). If you need them,
refer to the installer scripts under `tools/installers/`. After creating your
env with uv or pixi, you can run the installer scripts from `tools/`:

```bash
cd tools
./installers/<installer>.sh
```

Available installer scripts:

| Install | Description |
| --- | --- |
| [BeamformIt](https://github.com/xanguera/BeamformIt) | Beamforming tool for multi-channel speech enhancement. |
| [cauchy_mult (state-spaces)](https://github.com/HazyResearch/state-spaces) | Cauchy multiplication kernels for state-space models (S4). |
| [datasets](https://pypi.org/project/datasets/) | Hugging Face Datasets library. |
| [DeepXi](https://github.com/anicolson/DeepXi) | Speech enhancement toolkit. |
| [DiscreteSpeechMetrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | Metrics for discrete/unit-based speech models. |
| [fairscale](https://pypi.org/project/fairscale/) | Sharded training utilities for large models. |
| [fairseq](https://github.com/espnet/fairseq) | Sequence modeling toolkit used by some recipes. |
| [ffmpeg](https://ffmpeg.org/) | Audio/video IO and conversion. |
| [flash-attn](https://github.com/Dao-AILab/flash-attention) | Fast attention CUDA kernels. |
| [gss](https://github.com/desh2608/gss) | Guided source separation. |
| [gtn](https://pypi.org/project/gtn/) | Graph-based transducer networks library. |
| [ice-g2p](https://pypi.org/project/ice-g2p/) | Grapheme-to-phoneme for Icelandic. |
| [k2](https://k2-fsa.org/) | FSA toolkit for ASR/decoding. |
| [KenLM](https://github.com/kpu/kenlm) | N-gram language modeling toolkit. |
| [PyTorch Lightning](https://pypi.org/project/lightning/) | Training framework used by ESPnet3. |
| [Longformer](https://github.com/roshansh-cmu/longformer) | Long-context transformer model. |
| [loralib](https://pypi.org/project/loralib/) | LoRA adapters for fine-tuning. |
| [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) | Forced alignment for speech/text. |
| [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), [pytsmod](https://pypi.org/project/pytsmod/), [miditoolkit](https://pypi.org/project/miditoolkit/), [music21](https://pypi.org/project/music21/) | Music/singing TTS toolchain dependencies. |
| [mwerSegmenter](https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz) | Segmenter for mWER scoring. |
| [nkf](https://github.com/nurse/nkf) | Japanese text encoding conversion. |
| [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) | Face analysis and landmarks. |
| [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) | Neural vocoder. |
| [PESQ](https://github.com/LiChenda/itu_pesq) | Speech quality metric (PESQ). |
| [speech_tools](https://github.com/festvox/speech_tools), [festival](https://github.com/festvox/festival), [espeak-ng](https://github.com/espeak-ng/espeak-ng), [MBROLA](https://github.com/numediart/MBROLA) | Phonemization backends. |
| [py3mmseg](https://github.com/espnet/py3mmseg) | Japanese text segmentation. |
| [pyopenjtalk](https://pypi.org/project/pyopenjtalk/) | Japanese G2P / text frontend. |
| [RawNet](https://github.com/Jungjee/RawNet) | Speaker verification model. |
| [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech) | ReazonSpeech dataset/tools. |
| [s3prl](https://pypi.org/project/s3prl/) | Self-supervised speech representations. |
| [SCTK](https://github.com/usnistgov/SCTK) | Scoring toolkit (WER). |
| [SimulEval](https://github.com/facebookresearch/SimulEval) | Simultaneous translation evaluation. |
| [SpeechBrain](https://pypi.org/project/speechbrain/) | Speech toolkit (models/recipes). |
| [sph2pipe](https://github.com/burrmill/sph2pipe) | SPH to WAV conversion. |
| [tdmelodic_openjtalk](https://github.com/sarulab-speech/tdmelodic_openjtalk), [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) | Japanese singing TTS frontends. |
| [PyTorch](https://pytorch.org/) | Core deep learning framework. |
| [torch-optimizer](https://pypi.org/project/torch-optimizer/) | Extra optimizers for PyTorch. |
| [torcheval](https://pypi.org/project/torcheval/) | Metrics library for PyTorch. |
| [transformers](https://pypi.org/project/transformers/), [soxr](https://pypi.org/project/soxr/) | Transformers models and audio resampling. |
| [versa](https://github.com/shinjiwlab/versa) | VERSA toolkit (see repo for details). |
| [vidaug](https://github.com/okankop/vidaug) | Video data augmentation. |
| [visual deps](https://pypi.org/project/opencv-python/) | Visual/AV stack (OpenCV, ONNX, etc.). |
| [warp-transducer](https://github.com/ljn7/warp-transducer) | RNN-T CUDA extension. |
| [espnet/whisper](https://github.com/espnet/whisper) | ESPnet fork of Whisper for ASR. |

## Legacy conda setup

If you still rely on conda, the legacy setup script is available:

```bash
cd espnet/tools
. setup_anaconda.sh
```

Older guides may refer to this as `setup-conda.sh`. The workflow is the same,
but `setup_uv.sh` is recommended for faster, modern installs.

## CI-validated environments

The following environments are exercised in CI (`.github/workflows/`), which is
the current "known to work" matrix. This is not an exhaustive compatibility
guarantee, but a practical baseline.

| OS / runner | Python | PyTorch | Notes |
| --- | --- | --- | --- |
| Ubuntu (ubuntu-latest) | 3.10, 3.12 | 2.5.1, 2.7.1, 2.8.0, 2.9.1 | ESPnet2/3 unit + integration tests |
| Debian 12 (container) | 3.10 | 2.7.1 | ESPnet2/3 tests in a Debian container |
| macOS (macOS-latest) | 3.10 | 2.7.1 | Install check with and without conda |
| Windows (Windows-latest) | 3.10 | 2.7.1 | Install check |
