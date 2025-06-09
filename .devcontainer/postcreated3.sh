#!/usr/bin/env bash

processor=$1

echo "[INFO] install PyTorch for ${processor}"

if [ "${processor}" == "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision torchaudio
fi

pip install -e ".[all, test]"
