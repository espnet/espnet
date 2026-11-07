"""Tokenizer training text, selected by SQL from the omniio text dump."""

from __future__ import annotations

from pathlib import Path
from typing import List

from egs3.librispeech_100.esp2_asr_omniio.dataset.builder import resolve_dump_root
from espnet3.components.data.omniio_dataset import OmniIODataset

TEXT_QUERY = """
SELECT id, path AS text_path, start_byte AS text_start,
       end_byte - start_byte AS text_size
FROM read_parquet('{dump_root}/text/librispeech/{split}/metadata.parquet')
ORDER BY id
"""


def gather_training_text(
    recipe_dir: Path | None = None,
    dump_root: Path | None = None,
    split: str = "train-clean-100",
    query: str | None = None,
) -> List[str]:
    """Collect transcripts for tokenizer training from the omniio dump.

    Args:
        recipe_dir: Recipe root, used to find the dump when neither
            ``dump_root`` nor ``$OMNIIO_DUMP_ROOT`` is set.
        dump_root: omniio dump root.
        split: LibriSpeech split, filled into ``{split}``.
        query: DuckDB query returning ``id`` and ``text_path``/``text_start``/
            ``text_size`` (or an inline ``text`` column); defaults to every
            transcript of ``split``.

    Returns:
        Transcript strings for tokenizer training.
    """
    texts = OmniIODataset(
        query=query or TEXT_QUERY,
        fields={"text": "text"},
        params={"split": split},
        dump_root=resolve_dump_root(recipe_dir, dump_root),
    )
    return list(texts.iter_field("text"))
