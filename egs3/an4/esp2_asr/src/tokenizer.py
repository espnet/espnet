"""Training-only text for the original AN4 unigram tokenizer."""

from pathlib import Path

from egs3.an4.esp2_asr.dataset.builder import read_manifest


def gather_training_text(manifest_path):
    """Collect ASR training transcripts with their source repetition weights.

    Args:
        manifest_path: Prepared AN4 train TSV after source duration filtering.

    Returns:
        Transcripts in manifest order, including all speed variants.

    Raises:
        FileNotFoundError: create_dataset has not produced the manifest.
        ValueError: The manifest is empty or has duplicate utterance IDs.

    Examples:
        After the create_dataset stage:

        >>> texts = gather_training_text("data/manifest/train.tsv")
        >>> isinstance(texts[0], str)
        True
    """
    return [text for _, _, text in read_manifest(Path(manifest_path))]
