#!/usr/bin/env python3
"""Patch OWSM v4 checkpoint: add 3 new token embedding rows initialized to mean.

Usage:
    python local/init_new_tokens.py \
        --src_ckpt  ../../../../model_cache/owsm_v4_medium_1B/valid.loss.best.pth \
        --out_ckpt  ../../../../model_cache/owsm_v4_medium_1B_nahuatl/valid.loss.best.pth \
        --n_new     3
"""
import argparse
import os
import torch


def _extend_embedding(weight: torch.Tensor, n_new: int) -> torch.Tensor:
    """Extend a 2D [vocab, hidden] weight matrix by n_new rows (mean initialised)."""
    mean_row = weight.mean(dim=0, keepdim=True)
    new_rows = mean_row.expand(n_new, -1)
    return torch.cat([weight, new_rows], dim=0)


def _extend_bias(bias: torch.Tensor, n_new: int) -> torch.Tensor:
    """Extend a 1D [vocab] bias vector by n_new scalars (mean initialised)."""
    mean_val = bias.mean().expand(n_new)
    return torch.cat([bias, mean_val], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt', required=True)
    parser.add_argument('--out_ckpt', required=True)
    parser.add_argument('--n_new', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)), exist_ok=True)

    state = torch.load(args.src_ckpt, map_location='cpu')
    # state may be a plain dict or wrapped; unwrap if needed
    sd = state.get('model', state) if isinstance(state, dict) and 'model' in state else state

    embedding_keys = [
        k for k in sd
        if any(
            k.endswith(suffix)
            for suffix in (
                # Generic transformer patterns
                'embed_tokens.weight',
                'embed.weight',
                'lm_head.weight',
                'decoder.embed_tokens.weight',
                # OWSM v4 / ESPnet S2T patterns
                'decoder.embed.0.weight',
                'output_layer.weight',
                'output_layer.bias',
                'ctc_lo.weight',
                'ctc_lo.bias',
            )
        )
    ]
    if not embedding_keys:
        raise RuntimeError(
            f"No embedding keys found. Available keys (first 20): "
            f"{list(sd.keys())[:20]}"
        )

    for k in embedding_keys:
        original_shape = sd[k].shape
        if sd[k].dim() == 1:
            sd[k] = _extend_bias(sd[k], args.n_new)
        else:
            sd[k] = _extend_embedding(sd[k], args.n_new)
        print(f"  {k}: {original_shape} -> {sd[k].shape}")

    torch.save(state, args.out_ckpt)
    print(f"Saved patched checkpoint to {args.out_ckpt}")


if __name__ == '__main__':
    main()
