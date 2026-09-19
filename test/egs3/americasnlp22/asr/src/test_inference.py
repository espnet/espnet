"""Unit tests for the AmericasNLP 2022 inference output helper (no network)."""

import numpy as np

from egs3.americasnlp22.asr.src.inference import build_output


def test_build_output_uses_item_index_as_id() -> None:
    # Samples carry no identifier field; the index is the SCP row id.
    data = {"speech": np.zeros(4, dtype=np.float32), "text": "raw text 0"}
    output = build_output(data, [["hello world"]], 7)
    assert output == {
        "utt_id": "7",
        "hyp": "hello world",
        "ref": "raw text 0",
    }


def test_build_output_batched() -> None:
    data = [
        {"speech": np.zeros(4), "text": "raw text 0"},
        {"speech": np.zeros(4), "text": "raw text 1"},
    ]
    batch = build_output(data, [[["hyp one"]], [["hyp two"]]], [0, 1])
    assert [out["utt_id"] for out in batch] == ["0", "1"]
    assert [out["hyp"] for out in batch] == ["hyp one", "hyp two"]
    assert [out["ref"] for out in batch] == ["raw text 0", "raw text 1"]
