import numpy as np
from omniio import kaldi as kaldi_io

from espnet2.bin.tok_downstream_inference import build_iterator
from espnet2.tasks.gan_tok import GANSpeechTokenizerTask


def test_build_iterator_reads_tokens_and_selects_keys(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("utt1 1 2 3\nutt2 4 5\n", encoding="utf-8")
    key_file = tmp_path / "keys.scp"
    key_file.write_text("utt2 anything\n", encoding="utf-8")

    iterator = build_iterator(
        GANSpeechTokenizerTask,
        str(token_file),
        None,
        str(key_file),
        "float32",
        0,
        0,
    )
    keys, batch = next(iter(iterator))

    assert keys == ["utt2"]
    assert batch["speech"].tolist() == [[4, 5]]
    assert batch["speech_lengths"].tolist() == [2]


def test_build_iterator_reads_speaker_embeddings(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("utt1 1 2 3\n", encoding="utf-8")
    spembs_scp = tmp_path / "spembs.scp"
    with kaldi_io.WriteHelper(
        f"ark,scp:{tmp_path / 'spembs.ark'},{spembs_scp}"
    ) as writer:
        writer["utt1"] = np.array([0.1, 0.2], dtype=np.float32)

    iterator = build_iterator(
        GANSpeechTokenizerTask,
        str(token_file),
        str(spembs_scp),
        None,
        "float32",
        0,
        0,
    )
    keys, batch = next(iter(iterator))

    assert keys == ["utt1"]
    assert batch["speech"].tolist() == [[1, 2, 3]]
    np.testing.assert_allclose(batch["spembs"].numpy(), [[0.1, 0.2]])
