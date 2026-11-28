"""Tokenizer training text, selected by SQL from the omniio text dump."""

from __future__ import annotations

from pathlib import Path

from egs3.librispeech_100.esp2_asr.dataset.omniio.dataset import TEXT_QUERY
from espnet3.components.data.omniio_dataset import OmniIODataset


def gather_training_text(
    dump_root: Path | None = None,
    split: str = "train-clean-100",
    query: str | None = None,
) -> list[str]:
    """Collect transcripts for tokenizer training from the omniio dump.

    Args:
        dump_root: omniio dump root; defaults to ``$OMNIIO_DUMP_ROOT``.
        split: LibriSpeech split, bound to ``$split``.
        query: DuckDB query returning ``id`` and ``text_path``/``text_start``/
            ``text_size`` (or an inline ``text`` column); defaults to
            ``text_query`` in ``dataset/omniio/config.yaml``, every transcript
            of ``split``.

    Returns:
        Transcript strings for tokenizer training.

    Examples:
        >>> texts = gather_training_text(split="dev-clean")  # doctest: +SKIP
        >>> texts[0][:6]  # doctest: +SKIP
        'MISTER'
    """
    texts = OmniIODataset(
        query=query or TEXT_QUERY,
        fields={"text": "text"},
        params={"split": split},
        dump_root=dump_root,
    )
    return list(texts.iter_field("text"))
