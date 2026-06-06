"""Convert NVIDIA's HF Sortformer checkpoint into this ESPnet port.

The released ``nvidia/diar_sortformer_4spk-v1`` weights (``model.safetensors``,
🤗 Transformers ``SortformerOffline`` format) map to this implementation with a
near-identity key remap:

    fc_encoder.*          -> encoder.*
    tf_encoder.*          -> transformer_encoder.*
    sortformer_modules.*  -> sortformer_modules.*   (unchanged)
    tf_encoder.embed_positions.weight -> dropped (all-zero no-op in the export)

Buffers that this port computes deterministically (mel filterbank, STFT window)
are not present in the checkpoint and are left at their constructed values.

Usage:
    python -m sortformer.convert_hf_sortformer \\
        --hf_model nvidia/diar_sortformer_4spk-v1 \\
        --out sortformer_4spk.pth
"""

import argparse
from typing import Dict, Tuple

import torch

from .model import build_sortformer_model


def remap_key(key: str):
    """Return the ESPnet state-dict key for an HF key, or ``None`` to drop it."""
    if key == "tf_encoder.embed_positions.weight":
        return None  # all-zero no-op in the export
    if key.startswith("fc_encoder."):
        return "encoder." + key[len("fc_encoder.") :]
    if key.startswith("tf_encoder."):
        return "transformer_encoder." + key[len("tf_encoder.") :]
    if key.startswith("sortformer_modules."):
        return key
    return None


def convert_state_dict(
    hf_state: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], list, list]:
    """Map HF tensors to ESPnet keys; return (new_state, dropped, shape_mismatch)."""
    new_state = {}
    dropped = []
    mismatch = []
    for k, v in hf_state.items():
        nk = remap_key(k)
        if nk is None:
            dropped.append(k)
            continue
        if nk in model_state and tuple(model_state[nk].shape) != tuple(v.shape):
            mismatch.append((nk, tuple(v.shape), tuple(model_state[nk].shape)))
            continue
        new_state[nk] = v
    return new_state, dropped, mismatch


def load_hf_safetensors(path_or_repo: str) -> Dict[str, torch.Tensor]:
    """Load ``model.safetensors`` from a local path or a HF repo id."""
    import os

    from safetensors.torch import load_file

    if os.path.isdir(path_or_repo):
        path = os.path.join(path_or_repo, "model.safetensors")
    elif os.path.isfile(path_or_repo):
        path = path_or_repo
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=path_or_repo, filename="model.safetensors")
    return load_file(path)


def convert(hf_model: str, num_spk: int = 4):
    """Build the model and load the converted weights. Returns (model, report)."""
    model = build_sortformer_model(num_spk=num_spk)
    model_state = model.state_dict()
    hf_state = load_hf_safetensors(hf_model)
    new_state, dropped, mismatch = convert_state_dict(hf_state, model_state)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    # "missing" should only be deterministic buffers we recompute.
    report = dict(
        n_loaded=len(new_state),
        n_hf=len(hf_state),
        dropped=dropped,
        shape_mismatch=mismatch,
        missing=list(missing),
        unexpected=list(unexpected),
    )
    return model, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf_model", default="nvidia/diar_sortformer_4spk-v1")
    p.add_argument("--num_spk", type=int, default=4)
    p.add_argument("--out", required=True, help="output .pth path")
    args = p.parse_args()

    model, report = convert(args.hf_model, num_spk=args.num_spk)
    print(f"Loaded {report['n_loaded']}/{report['n_hf']} HF tensors.")
    if report["dropped"]:
        print("Dropped:", report["dropped"])
    if report["shape_mismatch"]:
        print("SHAPE MISMATCH:", report["shape_mismatch"])
    if report["unexpected"]:
        print("UNEXPECTED (HF keys not in model):", report["unexpected"])
    # Only deterministic buffers should be missing.
    allowed_missing = ("preprocessor.fb", "preprocessor.window")
    real_missing = [m for m in report["missing"] if not m.startswith(allowed_missing)]
    if real_missing:
        print("WARNING missing (not recomputed buffers):", real_missing)
    torch.save(model.state_dict(), args.out)
    print(f"Saved converted ESPnet checkpoint to {args.out}")


if __name__ == "__main__":
    main()
