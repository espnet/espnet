"""Tokenizer inputs for the SLURP SLU recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from egs3.slurp.slu.dataset.builder import (
    ensure_built,
    get_intents_path,
    get_manifest_path,
    read_manifest,
)
from egs3.slurp.slu.dataset.dataset import read_hypothesis_transcripts


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


def build_transcript_token_list(
    token_list_path: str | Path,
    recipe_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    splits: Iterable[str] = ("train", "train_synthetic"),
    transcript_source: str | Path | None = None,
) -> Path:
    """Write the word list the BERT post-decoder's transcript field is keyed by.

    ``ESPnetSLUModel`` turns the integer ``transcript`` field back into a string
    by looking each id up in ``transcript_token_list`` and joining with spaces,
    then hands that string to the Hugging Face tokenizer. So this list only has
    to round-trip words, and a plain word vocabulary of the training transcripts
    is what ``egs2/slurp/slu1`` builds (``--token_type word`` over
    ``data/train/transcript`` in stage 5 of ``slu.sh``).

    Called by :meth:`src.system.SLUSystem.train_tokenizer`, which reads its
    arguments from ``training_config.tokenizer.transcript_token_list``. Writing
    is skipped when the file is already there, so the stage stays re-runnable.

    Args:
        token_list_path: Where to write the list, one token per line.
        recipe_dir: Recipe directory used to locate the manifests. Defaults to
            the current working directory.
        source_dir: Optional corpus root override.
        splits: Manifest splits to collect words from.
        transcript_source: Where the transcripts come from, matching the
            ``data_src_args`` of the config being trained. ``None`` uses the
            corpus transcript, the ground-truth setting. A directory reads
            ``<transcript_source>/<split>/hyp_transcript.scp`` instead, so that
            the ASR-transcript config's vocabulary covers the words its
            first-pass model actually emits -- words only the reference uses
            would otherwise be unreachable, and hypothesis words outside the
            reference vocabulary would collapse to ``<unk>``.

    Returns:
        The path written, or the existing path when it was already there.

    Raises:
        FileNotFoundError: If ``transcript_source`` is set but the inference
            stage that fills it has not run for one of ``splits``.
        RuntimeError: If a hypothesis SCP does not line up with its manifest.
    """
    output_path = Path(token_list_path)
    if output_path.is_file():
        return output_path

    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    ensure_built(recipe_root, source_dir=source_dir)

    words: set[str] = set()
    for split in splits:
        rows = read_manifest(get_manifest_path(recipe_root, str(split)))
        if transcript_source is None:
            transcripts = [row["transcript"] for row in rows]
        else:
            transcripts = read_hypothesis_transcripts(
                Path(transcript_source) / str(split) / "hyp_transcript.scp",
                expected=len(rows),
            )
        for transcript in transcripts:
            words.update(transcript.split())

    tokens = ["<blank>", "<unk>", *sorted(words), "<sos/eos>"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path
