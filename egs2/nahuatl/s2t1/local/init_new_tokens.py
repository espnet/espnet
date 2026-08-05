#!/usr/bin/env python3
"""Patch OWSM v4 checkpoint: add N new token embedding rows initialized to mean.

Optionally also (a) generates an extended token list by reading the inline
token_list from the upstream model's config.yaml and appending the new tokens,
and (b) patches the SentencePiece BPE model so the new tokens are user-defined
symbols (tokenized as single pieces rather than split into subwords).

Usage:
    python local/init_new_tokens.py \
        --src_ckpt   ../../../../model_cache/owsm_v4_medium_1B/valid.loss.best.pth \
        --out_ckpt   ../../../../model_cache/owsm_v4_medium_1B_nahuatl/valid.loss.best.pth \
        --new_tokens "<nah_hid>" "<nah_ozg>" "<nah_ztp>" \
        --src_config ../../../../model_cache/owsm_v4_medium_1B/exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml \
        --out_token_list data/token_list_nahuatl.txt \
        --src_bpe    ../../../../model_cache/owsm_v4_medium_1B/data/token_list/bpe_unigram50000/bpe.model \
        --out_bpe    ../../../../model_cache/owsm_v4_medium_1B_nahuatl/data/token_list/bpe_unigram50000/bpe.model
"""

import argparse
import os

import torch


def _extend_embedding(weight: torch.Tensor, n_new: int) -> torch.Tensor:
    mean_row = weight.mean(dim=0, keepdim=True)
    new_rows = mean_row.expand(n_new, -1)
    return torch.cat([weight, new_rows], dim=0)


def _extend_bias(bias: torch.Tensor, n_new: int) -> torch.Tensor:
    mean_val = bias.mean().expand(n_new)
    return torch.cat([bias, mean_val], dim=0)


def extract_token_list(config_yaml: str) -> list[str]:
    """Extract the inline token_list from an ESPnet config.yaml.

    Uses PyYAML to correctly handle quoted and escaped tokens (e.g. ','  "\\x93").
    Reads line-by-line to avoid loading the full 50k-line file into memory at once.
    """
    import yaml

    tokens = []
    in_list = False
    with open(config_yaml, encoding="utf-8") as f:
        for line in f:
            if line.startswith("token_list:"):
                in_list = True
                continue
            if in_list:
                stripped = line.strip()
                if stripped.startswith("-"):
                    # Parse the YAML list item to handle quoting/escaping
                    token = yaml.safe_load(stripped[1:].strip() or "''")
                    tokens.append(str(token) if token is not None else "")
                else:
                    break  # next top-level key
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_ckpt", required=True)
    parser.add_argument("--out_ckpt", required=True)
    # New tokens as positional-style repeated flag
    parser.add_argument(
        "--new_tokens",
        nargs="+",
        default=["<nah_hid>", "<nah_ozg>", "<nah_ztp>"],
        help="New tokens to append (default: 3 Nahuatl region tokens)",
    )
    # Token list generation (optional)
    parser.add_argument(
        "--src_config",
        default=None,
        help="Path to upstream model config.yaml with inline token_list",
    )
    parser.add_argument(
        "--out_token_list", default=None, help="Where to write the extended token list"
    )
    parser.add_argument(
        "--src_bpe", default=None, help="Path to upstream SentencePiece bpe.model"
    )
    parser.add_argument(
        "--out_bpe",
        default=None,
        help="Where to write the patched bpe.model (new tokens as user-defined symbols)",
    )
    args = parser.parse_args()

    n_new = len(args.new_tokens)

    # ── Patch checkpoint ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)), exist_ok=True)

    state = torch.load(args.src_ckpt, map_location="cpu")
    sd = (
        state.get("model", state)
        if isinstance(state, dict) and "model" in state
        else state
    )

    embedding_keys = [
        k
        for k in sd
        if any(
            k.endswith(suffix)
            for suffix in (
                "embed_tokens.weight",
                "embed.weight",
                "lm_head.weight",
                "decoder.embed_tokens.weight",
                "decoder.embed.0.weight",
                "output_layer.weight",
                "output_layer.bias",
                "ctc_lo.weight",
                "ctc_lo.bias",
            )
        )
    ]
    if not embedding_keys:
        raise RuntimeError(
            f"No embedding keys found. Available keys (first 20): {list(sd.keys())[:20]}"
        )

    for k in embedding_keys:
        original_shape = sd[k].shape
        if sd[k].dim() == 1:
            sd[k] = _extend_bias(sd[k], n_new)
        else:
            sd[k] = _extend_embedding(sd[k], n_new)
        print(f"  {k}: {original_shape} -> {sd[k].shape}")

    torch.save(state, args.out_ckpt)
    print(f"Saved patched checkpoint to {args.out_ckpt}")

    # ── Generate extended token list ────────────────────────────────────────
    if args.src_config and args.out_token_list:
        print(f"Extracting token list from {args.src_config} ...")
        tokens = extract_token_list(args.src_config)
        if not tokens:
            raise RuntimeError(f"No inline token_list found in {args.src_config}")
        print(f"  Found {len(tokens)} tokens in upstream config")
        tokens.extend(args.new_tokens)
        os.makedirs(
            os.path.dirname(os.path.abspath(args.out_token_list)) or ".", exist_ok=True
        )
        with open(args.out_token_list, "w", encoding="utf-8") as f:
            f.write("\n".join(tokens) + "\n")
        print(f"Wrote {len(tokens)} tokens to {args.out_token_list}")

    # ── Patch the BPE model ─────────────────────────────────────────────────
    # Add the new tokens as SentencePiece user-defined symbols so they tokenize
    # as single pieces. Without this the tokenizer splits e.g. "<nah_hid>" into
    # subwords and the new embeddings above are never used.
    if args.src_bpe and args.out_bpe:
        from sentencepiece import sentencepiece_model_pb2 as pb2

        print(f"Patching BPE model from {args.src_bpe} ...")
        proto = pb2.ModelProto()
        with open(args.src_bpe, "rb") as f:
            proto.ParseFromString(f.read())
        existing = {p.piece for p in proto.pieces}
        n_before = len(proto.pieces)
        for sym in args.new_tokens:
            if sym in existing:
                continue
            piece = proto.SentencePiece()
            piece.piece = sym
            piece.score = 0.0
            piece.type = pb2.ModelProto.SentencePiece.USER_DEFINED
            proto.pieces.append(piece)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_bpe)) or ".", exist_ok=True)
        with open(args.out_bpe, "wb") as f:
            f.write(proto.SerializeToString())
        print(f"  SentencePiece pieces {n_before} -> {len(proto.pieces)}")
        print(f"Wrote patched BPE model to {args.out_bpe}")


if __name__ == "__main__":
    main()
