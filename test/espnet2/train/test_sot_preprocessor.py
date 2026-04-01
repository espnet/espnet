import numpy as np
import pytest

pytest.importorskip("whisper")

from espnet2.train.sot_preprocessor import SOTWhisperPreprocessor


@pytest.fixture(scope="module")
def preprocessor():
    return SOTWhisperPreprocessor(
        train=True,
        whisper_language="en",
        whisper_task="transcribe",
    )


@pytest.mark.timeout(30)
def test_preprocessor_init(preprocessor):
    assert preprocessor.vocab_size > 51865
    assert len(preprocessor.prefix_ids) == 2  # [<|en|>, <|transcribe|>]


@pytest.mark.timeout(30)
def test_preprocessor_basic_text(preprocessor):
    data = {"text": "hello world <|endoftext|>"}
    out = preprocessor("utt1", data)
    assert isinstance(out["text"], np.ndarray)
    assert out["text"].dtype == np.int64
    # Starts with prefix [<|en|>, <|transcribe|>]
    assert out["text"][0] == preprocessor.prefix_ids[0]
    assert out["text"][1] == preprocessor.prefix_ids[1]


@pytest.mark.timeout(30)
def test_preprocessor_timestamps(preprocessor):
    data = {"text": "<|0.00|> hello<|1.00|> <|endoftext|>"}
    out = preprocessor("utt1", data)
    ids = out["text"].tolist()
    # Timestamp token IDs are in range 50364-51864
    ts_ids = [i for i in ids if 50364 <= i <= 51864]
    assert len(ts_ids) >= 2  # at least <|0.00|> and <|1.00|>


@pytest.mark.timeout(30)
def test_preprocessor_speaker_change(preprocessor):
    data = {"text": "hello <sc> world <|endoftext|>"}
    out = preprocessor("utt1", data)
    ids = out["text"].tolist()
    # <sc> should be tokenized as a single token (ID >= 51865)
    assert any(i >= 51865 for i in ids)


@pytest.mark.timeout(30)
def test_preprocessor_added_tokens_from_file(tmp_path):
    tokens_file = tmp_path / "tokens.txt"
    tokens_file.write_text("<sc>\n")
    prep = SOTWhisperPreprocessor(
        train=True,
        whisper_language="en",
        whisper_task="transcribe",
        added_tokens_txt=str(tokens_file),
    )
    assert "<sc>" in prep.added_tokens


@pytest.mark.timeout(30)
def test_preprocessor_output_deterministic(preprocessor):
    data1 = {
        "text": "<|0.00|> hello world<|1.00|> <sc> <|0.50|> hi<|1.50|> <|endoftext|>"
    }
    data2 = {
        "text": "<|0.00|> hello world<|1.00|> <sc> <|0.50|> hi<|1.50|> <|endoftext|>"
    }
    out1 = preprocessor("utt1", data1)
    out2 = preprocessor("utt1", data2)
    assert np.array_equal(out1["text"], out2["text"])


@pytest.mark.timeout(30)
def test_preprocessor_existing_bpe_token(tmp_path):
    """Test that ???? maps to its existing BPE ID (25629), not a new token."""
    tokens_file = tmp_path / "tokens.txt"
    tokens_file.write_text("????\n")
    prep = SOTWhisperPreprocessor(
        train=True,
        whisper_language="en",
        whisper_task="transcribe",
        added_tokens_txt=str(tokens_file),
    )
    assert "????" in prep.added_tokens
    assert prep.added_token_map["????"] == 25629
    assert prep.vocab_size == prep.encoding.n_vocab  # no expansion

    data = {"text": "<|0.00|> hello<|1.00|> ???? <|0.50|> world<|1.50|> <|endoftext|>"}
    out = prep("utt1", data)
    assert 25629 in out["text"].tolist()
