"""Generate text_shape.bpe / src_text_shape.bpe to match egs2's batching.

espnet3's collect_stats emits only feats/feats_lengths (see
``exp/stats/train/stats_keys``), but ``egs2/must_c/st1/st.sh`` passes
``--train_shape_file`` THREE times (st.sh:1345-1349)::

    speech_shape
    text_shape.${tgt_token_type}
    src_text_shape.${src_token_type}      # use_src_lang=true for this recipe

``NumElementsBatchSampler`` sums ``len(batch) * max_len * feat_dim`` over every
shape file it is given, so with ``feats_shape`` alone the two text streams are
invisible to the batcher. That is not a rounding error: a batch of very short
utterances is then allowed to grow without limit, and batch 0 of this recipe
reached 1,732 utterances and died in the encoder feed-forward with::

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB.
    GPU 0 has a total capacity of 39.49 GiB of which 715.25 MiB is free.

on an otherwise idle 40 GB A100. Measured, not predicted.

The files are 2-D ("L,V"), exactly as st.sh:1249-1254 writes them by appending
the vocabulary size, so one target token costs V bins and the text terms
dominate short-utterance batches the same way they do in egs2.

Keys are the dataset row index, matching what collect_stats writes into
feats_shape, and all three files must describe the same key set -- so this
walks the FILTERED dataset (st.sh stage 4 bounds; see dataset/config.yaml) and
applies the same per-side case conventions the preprocessor will apply.

    python gen_text_shape.py
"""

import pathlib

import numpy as np
import sentencepiece as spm
from datasets import load_from_disk

from egs3.must_c.st.dataset import Dataset
from espnet3.systems.st.text_case import apply_case

RECIPE_DIR = pathlib.Path(__file__).resolve().parent
CACHE = RECIPE_DIR / "data" / "hf" / "en_de" / "hf_audio_index"
TGT_BPE = RECIPE_DIR / "data" / "bpe_tgt_4000"
SRC_BPE = RECIPE_DIR / "data" / "bpe_src_4000"
# (logical split, stats subdir), mirroring conf/tuning/train_st_conformer.yaml
SPLITS = (("train", "train"), ("dev", "valid"))


def _vocab_size(bpe_dir: pathlib.Path) -> int:
    """Vocabulary size as st.sh takes it: the line count of the token list."""
    return sum(1 for _ in (bpe_dir / "tokens.txt").open(encoding="utf-8"))


def _lengths(texts, model_file, chunk=100_000):
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    return np.fromiter(
        (
            len(ids)
            for start in range(0, len(texts), chunk)
            for ids in sp.encode(texts[start : start + chunk], num_threads=16)
        ),
        dtype=np.int32,
        count=len(texts),
    )


def main() -> None:
    tgt_v, src_v = _vocab_size(TGT_BPE), _vocab_size(SRC_BPE)
    print(f"vocab sizes: tgt={tgt_v} src={src_v}")
    cache_cfg = {
        "enabled": True,
        "backend": "hf",
        "cache_dir": str(RECIPE_DIR / "data" / "hf" / "en_de"),
    }

    for split, stats_split in SPLITS:
        # _keep is the filter's surviving positions; None means unfiltered.
        ds = Dataset(
            split=split,
            recipe_dir=str(RECIPE_DIR),
            source_dir=str(RECIPE_DIR / "data"),
            cache=cache_cfg,
            task="st",
            tgt_lang="de",
        )
        raw = load_from_disk(str(CACHE / split))
        keep = ds._keep if ds._keep is not None else range(len(raw))

        src_raw, tgt_raw = raw["src_text"], raw["tgt_text"]
        # Same case conventions the preprocessor will see (egs2 run.sh:
        # src_case=lc.rm, tgt_case=tc).
        tgt_texts = [apply_case(str(tgt_raw[i]), "tc") for i in keep]
        src_texts = [apply_case(str(src_raw[i]), "lc.rm") for i in keep]
        assert len(tgt_texts) == len(ds), (len(tgt_texts), len(ds))

        for name, texts, model, vocab in (
            ("text_shape.bpe", tgt_texts, TGT_BPE / "bpe.model", tgt_v),
            ("src_text_shape.bpe", src_texts, SRC_BPE / "bpe.model", src_v),
        ):
            lengths = _lengths(texts, model)
            out = RECIPE_DIR / "exp" / "stats" / stats_split / name
            with out.open("w") as stream:
                for index, length in enumerate(lengths):
                    stream.write(f"{index} {length},{vocab}\n")
            pct = [np.percentile(lengths, p) for p in (50, 99, 100)]
            print(
                f"[{stats_split}] {len(lengths):,} rows -> {out.name}  "
                f"p50={pct[0]:.0f} p99={pct[1]:.0f} max={pct[2]:.0f} tokens"
            )


if __name__ == "__main__":
    main()
