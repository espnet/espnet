"""Check ordered, lazy LM text access through DataOrganizer."""

import pickle

import pytest

from espnet3.systems.esp2_asr.lm_dataset import LMTextDataset


def test_text_order_unicode_and_worker_serialization(tmp_path):
    """Keep file order and UTF-8 text across dataloader serialization."""
    path = tmp_path / "text"
    path.write_text("z HELLO 世界\nempty\na AGAIN\n", encoding="utf-8")
    dataset = pickle.loads(pickle.dumps(LMTextDataset(path)))
    assert len(dataset) == 2
    assert dataset[0] == {"text": "HELLO 世界"}
    assert dataset[1] == {"text": "AGAIN"}
    with pytest.raises(IndexError):
        dataset[2]


def test_empty_text_is_rejected(tmp_path):
    """Reject an empty corpus before training starts."""
    path = tmp_path / "text"
    path.write_text("empty\n\n")
    with pytest.raises(ValueError, match="nonempty"):
        LMTextDataset(path)
