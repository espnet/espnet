"""Tokenizer inputs for the SLURP SLU recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from egs3.slurp.esp2_slu.dataset.builder import (
    ensure_built,
    get_intents_path,
    get_manifest_path,
    read_manifest,
)
from egs3.slurp.esp2_slu.dataset.dataset import read_hypothesis_transcripts


def gather_training_text(
    recipe_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    splits: Iterable[str] = ("train", "train_synthetic"),
) -> List[str]:
    """Collect the transcripts SentencePiece is trained on.

    Wired in from ``training.yaml`` as ``tokenizer.text_builder.func`` and
    called by the ``train_tokenizer`` stage. Only the transcript column is
    returned: intent labels enter the vocabulary as reserved symbols instead
    (see :func:`read_intent_labels`), which keeps each ``scenario_action`` label
    a single token the way ``egs2/slurp/asr1`` does with word-level intents.

    Args:
        recipe_dir: Recipe directory used to locate the manifests. Defaults to
            the current working directory, which is the recipe root when
            ``run.py`` is invoked as documented.
        source_dir: Optional corpus root override, forwarded to the builder if
            the manifests still have to be built.
        splits: Manifest splits to read.

    Returns:
        One transcript per training utterance, in manifest order.

    Raises:
        RuntimeError: If the selected splits hold no transcript text, which
            means ``create_dataset`` produced empty manifests.
    """
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    ensure_built(recipe_root, source_dir=source_dir)

    texts: list[str] = []
    for split in splits:
        for row in read_manifest(get_manifest_path(recipe_root, str(split))):
            if row["transcript"]:
                texts.append(row["transcript"])

    if not texts:
        raise RuntimeError("No transcript text found for tokenizer training.")
    return texts


def read_intent_labels(recipe_dir: str | Path) -> List[str]:
    """Read the intent labels written next to the manifests by ``create_dataset``.

    Args:
        recipe_dir: Recipe directory.

    Returns:
        The sorted intent labels, one per line of ``data/manifest/intents.txt``.

    Raises:
        FileNotFoundError: If the list is missing, meaning ``create_dataset``
            has not run yet.
        RuntimeError: If the file exists but is empty.
    """
    intents_path = get_intents_path(recipe_dir)
    if not intents_path.is_file():
        raise FileNotFoundError(
            f"Intent label list not found: {intents_path}. "
            "Run `python run.py --stages create_dataset` first."
        )
    labels = [
        line.strip()
        for line in intents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise RuntimeError(f"Intent label list is empty: {intents_path}")
    return labels


def gather_transcript_text(
    recipe_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    splits: Iterable[str] = ("train", "train_synthetic"),
    transcript_source: str | Path | None = None,
) -> List[str]:
    """Collect the transcripts the BERT post-decoder's token list is built from.

    Wired in from a training config as
    ``tokenizer.transcript_token_list.text_builder.func``, and called by
    ``Esp2SluSystem``, which turns these lines into the word vocabulary.
    ``ESPnetSLUModel`` detokenizes the transcript field through that list before
    handing it to the Hugging Face tokenizer, so a word missing from it is a
    word the post-decoder can never read.

    Args:
        recipe_dir: Recipe directory used to locate the manifests. Defaults to
            the current working directory.
        source_dir: Optional corpus root override.
        splits: Manifest splits to read.
        transcript_source: Where the transcripts come from, matching the
            ``data_src_args`` of the config being trained. ``None`` uses the
            corpus transcript, the ground-truth setting. A directory reads
            ``<transcript_source>/<split>/hyp_transcript.scp`` instead, so that
            the ASR-transcript config's vocabulary covers the words its
            first-pass model actually emits -- words only the reference uses
            would otherwise be unreachable, and hypothesis words outside the
            reference vocabulary would collapse to ``<unk>``.

    Returns:
        One transcript per utterance, in manifest order.

    Raises:
        FileNotFoundError: If ``transcript_source`` is set but the inference
            stage that fills it has not run for one of ``splits``.
        RuntimeError: If a hypothesis SCP does not line up with its manifest.
    """
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    ensure_built(recipe_root, source_dir=source_dir)

    texts: list[str] = []
    for split in splits:
        rows = read_manifest(get_manifest_path(recipe_root, str(split)))
        if transcript_source is None:
            texts.extend(row["transcript"] for row in rows)
        else:
            texts.extend(
                read_hypothesis_transcripts(
                    Path(transcript_source) / str(split) / "hyp_transcript.scp",
                    expected=len(rows),
                )
            )
    return texts
