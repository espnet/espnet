"""Convert NVIDIA's HF Sortformer checkpoint into this ESPnet port.

The released ``nvidia/diar_sortformer_4spk-v1`` weights (``model.safetensors``,
in the Hugging Face Transformers ``SortformerOffline`` format) map onto this
implementation with a near-identity key remap:

    fc_encoder.*          -> encoder.*
    tf_encoder.*          -> transformer_encoder.*
    sortformer_modules.*  -> sortformer_modules.*   (unchanged)
    tf_encoder.embed_positions.weight -> dropped (all-zero no-op in the export)

Buffers that this port computes deterministically (mel filterbank, STFT window)
are not present in the checkpoint and are left at their constructed values.

This module is meant to be run as a CLI to write a ready-to-load ``.pth``::

    python -m espnet3.systems.diar.sortformer.convert_hf_sortformer \\
        --hf_model nvidia/diar_sortformer_4spk-v1 \\
        --out sortformer_4spk.pth

It can also be used programmatically via :func:`convert`, which returns the
loaded model together with a report describing which tensors were transferred.
"""

import argparse
from typing import Dict, Tuple

import torch

from .model import build_sortformer_model


def remap_key(key: str):
    """Translate one HF state-dict key into this port's naming.

    Args:
        key: A key from the HF ``model.safetensors`` state dict.

    Returns:
        The corresponding ESPnet state-dict key, or ``None`` if the tensor
        should be dropped (an unknown key, or the all-zero positional embedding
        that the HF export keeps but this port does not use).
    """
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
    """Remap a whole HF state dict onto this port's keys.

    Each HF tensor is renamed with :func:`remap_key`. Tensors that cannot be
    mapped, or whose shape does not match the target model, are recorded instead
    of being loaded.

    Args:
        hf_state: The HF source state dict (key -> tensor).
        model_state: ``model.state_dict()`` of the target ESPnet model, used to
            validate target keys and shapes.

    Returns:
        A tuple ``(new_state, dropped, shape_mismatch)`` where ``new_state`` maps
        ESPnet keys to tensors ready to load, ``dropped`` lists HF keys with no
        target, and ``shape_mismatch`` lists ``(key, hf_shape, model_shape)``
        triples for keys whose shapes disagree.
    """
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
    """Load the HF ``model.safetensors`` state dict.

    Args:
        path_or_repo: A local directory containing ``model.safetensors``, a
            direct path to the file, or a Hugging Face repo id (e.g.
            ``"nvidia/diar_sortformer_4spk-v1"``) to download from the Hub.

    Returns:
        The loaded state dict (key -> tensor).
    """
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
    """Build an ESPnet Sortformer model and load the converted HF weights.

    Args:
        hf_model: Local path or HF repo id passed to :func:`load_hf_safetensors`.
        num_spk: Number of speakers the model is built for (4 for the released
            offline checkpoint).

    Returns:
        A tuple ``(model, report)``. ``model`` is the built model with the
        converted weights loaded (non-strict). ``report`` is a dict with counts
        and diagnostics: ``n_loaded``, ``n_hf``, ``dropped``, ``shape_mismatch``,
        ``missing`` (keys absent from the source; should be only the recomputed
        mel/STFT buffers) and ``unexpected``.

    Example:
        >>> model, report = convert("nvidia/diar_sortformer_4spk-v1")
        >>> report["n_loaded"], report["shape_mismatch"]
        (..., [])
    """
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
    """CLI entrypoint: convert the HF checkpoint and save it as a ``.pth``.

    Parses ``--hf_model``, ``--num_spk`` and the required ``--out`` path, runs
    :func:`convert`, prints the conversion report (warning only for genuinely
    missing tensors, i.e. anything other than the recomputed buffers), and writes
    the resulting state dict to ``--out``.
    """
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
