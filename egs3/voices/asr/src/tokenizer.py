"""Original, pre-filtering VOiCES transcripts for SentencePiece."""

from pathlib import Path


def gather_training_text(text_path):
    """Read the source recipe's text before ASR duration filtering.

    Args:
        text_path: Builder-produced tokenizer_train.txt without utterance IDs.

    Returns:
        Ordered transcripts, preserving repetitions across recording variants.

    Raises:
        FileNotFoundError: The create_dataset stage has not produced the text.
    """
    return Path(text_path).read_text(encoding="utf-8").splitlines()
