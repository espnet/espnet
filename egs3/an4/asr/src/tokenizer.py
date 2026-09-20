"""Training-only text for the original AN4 unigram tokenizer."""

from pathlib import Path

from egs3.an4.asr.dataset.builder import read_manifest


def gather_training_text(manifest_path):
    """Collect ASR training transcripts with their source repetition weights.

    Args:
        manifest_path: Prepared AN4 train TSV after source duration filtering.

    Returns:
        Transcripts in manifest order, including all speed variants.

    Raises:
        FileNotFoundError: create_dataset has not produced the manifest.
        ValueError: The manifest is empty or has duplicate utterance IDs.
    """
    return [text for _, _, text in read_manifest(Path(manifest_path))]
