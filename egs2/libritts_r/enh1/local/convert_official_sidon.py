#!/usr/bin/env python3
"""Convert the official Sidon LoRA adapter into an ESPnet-Sidon checkpoint.

The official release publishes the feature predictor's adapter as real
weights (``sarulab-speech/sidon_raw_weight``, MIT), and its key layout turns
out to be identical to this recipe's model apart from a mechanical prefix:

    official   base_model.model.encoder.layers.N.ffn1.output_dense.lora_A.weight
    ESPnet     ssl_encoder.student.encoder.layers.N.ffn1.output_dense.lora_A.default.weight

Same module path, same 32 tensors (8 layers x {ffn1, ffn2} x {A, B}), same
shapes -- (64, 4096) and (1024, 64), i.e. rank 64 on the FFN output
projections. So the conversion is a rename, not a reimplementation, and the
fact that it round-trips is itself the strongest available check that this
recipe reproduces the published architecture rather than merely resembling it.

Two uses:

  * run official Sidon through the ESPnet inference path, so it can be scored
    by the same harness as everything else instead of a separate script
  * warm-start a derived model from the published predictor rather than from
    scratch

Usage
-----
python local/convert_official_sidon.py \
    --adapter  <hf snapshot>/adapter_model.safetensors \
    --out      exp/official_sidon/valid.loss.best.pth
"""

import argparse
import os

import torch


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        required=True,
        help="adapter_model.safetensors from sarulab-speech/sidon_raw_weight",
    )
    parser.add_argument("--out", required=True, help="output .pth for --model_file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless every official tensor is consumed",
    )
    return parser


def convert_keys(state):
    """Rename official adapter keys to the ESPnet module layout."""
    out = {}
    unmapped = []
    for key, value in state.items():
        if not key.startswith("base_model.model."):
            unmapped.append(key)
            continue
        tail = key[len("base_model.model.") :]
        # peft stores the active adapter under its name; the recipe builds the
        # default adapter, so ".weight" becomes ".default.weight".
        if tail.endswith(".weight") and (".lora_A" in tail or ".lora_B" in tail):
            tail = tail[: -len(".weight")] + ".default.weight"
        else:
            unmapped.append(key)
            continue
        out[f"ssl_encoder.student.{tail}"] = value
    return out, unmapped


def main():
    args = get_parser().parse_args()
    from safetensors.torch import load_file

    state = load_file(args.adapter)
    converted, unmapped = convert_keys(state)

    print(f"official tensors : {len(state)}")
    print(f"converted        : {len(converted)}")
    print(f"unmapped         : {len(unmapped)}")
    for key in unmapped[:5]:
        print(f"    {key}")
    if unmapped and args.strict:
        raise SystemExit("unmapped tensors present and --strict was given")
    if not converted:
        raise SystemExit("nothing converted; is this the right adapter file?")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(converted, args.out)
    total = sum(v.numel() for v in converted.values())
    print(f"wrote {args.out}  ({total / 1e6:.2f}M parameters)")
    print(
        "Load with enh_inference_sidon.py --model_file <this> and the official "
        "decoder via --sidon_vocoder. Note the recipe's train.yaml must keep "
        "lora_rank=64 for the shapes to line up."
    )


if __name__ == "__main__":
    main()
