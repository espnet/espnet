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
pip install -r espnet2/speechlm/requirement.txt
```

Alternatively, install ESPnet with its SpeechLM extra:

```bash
pip install -e ".[speechlm]"
```

## 3. Install FlashAttention

Choose the backend for your GPU; see
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
