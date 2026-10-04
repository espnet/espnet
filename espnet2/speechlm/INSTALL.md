# ESPnet SpeechLM installation

SpeechLM can be installed directly with the quick steps below; following the
full ESPnet installation procedure is not required.

Activate a fresh Python environment (conda or venv). Run the commands below
from the ESPnet repository root.

## 1. Install PyTorch

Install `torch`, `torchaudio`, and `torchvision` using the
[PyTorch installer](https://pytorch.org/get-started/locally/), choosing versions
and a CUDA build suitable for your environment.

For example, with CUDA 12.8 wheels:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 2. Install SpeechLM dependencies

```bash
pip install -e ".[speechlm]"
```

Flash Linear Attention (FLA) and TileLang have prebuilt wheels for supported
Linux platforms and do not require `nvcc` to install. Their GPU kernels still
need a supported accelerator at runtime.

## 3. Install CUDA extensions

Install these extensions separately on Linux with a CUDA toolkit (`nvcc`)
and a C++ compiler compatible with the CUDA build of PyTorch chosen above.
Use `--no-build-isolation` so each build uses the installed PyTorch instead
of resolving another version in an isolated build environment.

For models that use `causal-conv1d`, install its build tools and the extension:

```bash
pip install wheel ninja
pip install --no-build-isolation "causal-conv1d>=1.6.2"
```

For FlashAttention, choose the backend for your GPU; see
[FlashAttention on GitHub](https://github.com/Dao-AILab/flash-attention).
For example, install **FlashAttention 3 on H100/H800** from `hopper/`:

```bash
pip install --no-build-isolation \
    'git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper'
```

For this example, set `attn_implementation` to `flash_attention_3`
for the language model and audio encoder.

If no matching wheel exists, pip may compile CUDA extensions locally, requiring
a CUDA toolkit (`nvcc`) and C++ compiler.
