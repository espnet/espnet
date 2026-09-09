#!/usr/bin/env python3
"""Convert the published Sidon vocoder into an ESPnet-Sidon checkpoint.

The release (``sarulab-speech/sidon-v0.1``, MIT) ships the vocoder only as a
frozen TorchScript graph, ``decoder_{cpu,cuda}.pt``. That is enough to run
it but not to train from it: a frozen graph has no parameters. This script
recovers the weights into the recipe's ``SidonVocoder`` -- the same DAC
decoder, so the recovery is exact and is verified by running both on the
same input -- and writes them in the layout of a stage-7/8 checkpoint.

Two uses:

  * warm-start stage 8 (finetune on predicted features) from the published
    vocoder instead of a stage-7 pretrain:
        --init_param exp/official_sidon_vocoder/vocoder.pth:vocoder:vocoder
  * run the official vocoder through the same inference path as a trained
    one (--vocoder_train_config / --vocoder_model_file), as a check that
    the path reproduces --sidon_vocoder exactly.

Usage
-----
python local/convert_official_sidon_vocoder.py \
    --torchscript <hf snapshot>/decoder_cuda.pt \
    --out_dir exp/official_sidon_vocoder
"""

import argparse
import os

import torch
import yaml
from torch import nn
from torch.nn.utils import weight_norm

from espnet2.enh.decoder.sidon_vocoder import SidonVocoder

VOCODER_CONF = {"channels": 1536, "rates": [8, 5, 4, 3, 2]}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torchscript", required=True, help="decoder_cpu.pt or decoder_cuda.pt"
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_dim", type=int, default=1024)
    return parser


def main():
    args = get_parser().parse_args()
    vocoder = SidonVocoder(input_dim=args.input_dim, **VOCODER_CONF)
    vocoder.load_official_torchscript(args.torchscript, verify=True)
    # Training checkpoints carry weight-normalised convolutions
    # (weight_g / weight_v); re-parametrise so the file loads into a fresh
    # training model with --init_param without any key surgery.
    for module in vocoder.modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            weight_norm(module)
    state = {f"vocoder.{k}": v for k, v in vocoder.state_dict().items()}

    # Round trip through a fresh model, then compare with the graph again.
    check = SidonVocoder(input_dim=args.input_dim, **VOCODER_CONF)
    check.load_state_dict(
        {k[len("vocoder.") :]: v for k, v in state.items()}, strict=True
    )
    check.remove_weight_norm()
    check.eval()
    scripted = torch.jit.load(args.torchscript, map_location="cpu")
    torch.manual_seed(0)
    x = torch.randn(2, args.input_dim, 40)
    with torch.no_grad():
        err = (check(x) - scripted(x)).abs().max().item()
    if err > 1e-4:
        raise RuntimeError(
            f"round trip does not reproduce the official decoder ({err:.2e})"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(state, os.path.join(args.out_dir, "vocoder.pth"))
    with open(os.path.join(args.out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "vocoder_conf": VOCODER_CONF,
                "input_sr": 16000,
                "output_sr": 48000,
                "source": os.path.abspath(args.torchscript),
            },
            f,
        )
    n = sum(
        v.numel() for k, v in state.items() if k.endswith(("weight_v", "bias", "alpha"))
    )
    print(
        f"wrote {args.out_dir}/vocoder.pth ({len(state)} tensors, "
        f"{n / 1e6:.1f}M parameters); max abs error vs TorchScript {err:.1e}"
    )


if __name__ == "__main__":
    main()
