"""MuST-C sub-dataset, checked against a synthetic corpus.

What is pinned here is the part the dump cannot settle: every language pair
regenerates the English side, so the ASR utterances must be collapsed to one
copy per id, and the survivor must be chosen deterministically.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import yaml

from egs3.owsm_v4.owsm.dataset.sub_datasets.must_c import Dataset, DatasetBuilder

RATE = 16000


def _corpus(tmp_path, pairs, talk_seconds=60.0):
    """Write a MuST-C-shaped tree.

    ``pairs`` maps a language to a list of
    ``(wav, offset, duration, source, target)``.
    """
    root = tmp_path / "must-c"
    for lang, segments in pairs.items():
        data = root / f"en-{lang}" / "data" / "dev"
        (data / "txt").mkdir(parents=True, exist_ok=True)
        (data / "wav").mkdir(parents=True, exist_ok=True)
        for wav in {segment[0] for segment in segments}:
            sf.write(
                str(data / "wav" / wav),
                np.zeros(int(talk_seconds * RATE), dtype="int16"),
                RATE,
                subtype="PCM_16",
            )
        entries = [
            {"wav": wav, "offset": offset, "duration": duration, "speaker_id": "spk.1"}
            for wav, offset, duration, _, _ in segments
        ]
        (data / "txt" / "dev.yaml").write_text(yaml.safe_dump(entries), "utf-8")
        (data / "txt" / "dev.en").write_text(
            "\n".join(s[3] for s in segments) + "\n", "utf-8"
        )
        (data / "txt" / f"dev.{lang}").write_text(
            "\n".join(s[4] for s in segments) + "\n", "utf-8"
        )
        # train is never read by these tests; it only satisfies the source check.
        train = root / f"en-{lang}" / "data" / "train" / "txt"
        train.mkdir(parents=True, exist_ok=True)
        (train / "train.yaml").write_text("[]\n", "utf-8")
    return root


def _rows(root, **kwargs):
    # The builder defaults to all 14 configured pairs; a fixture has two.
    kwargs.setdefault("languages", sorted(p.name[3:] for p in root.glob("en-*")))
    return list(DatasetBuilder().iter_rows(root, "dev", **kwargs))


def _two_pairs(tmp_path):
    return _corpus(
        tmp_path,
        {
            "de": [("t1.wav", 0.0, 2.0, "hello there", "hallo da")],
            "fr": [("t1.wav", 0.0, 2.0, "hello there", "bonjour la")],
        },
    )


def test_each_segment_yields_one_asr_and_one_st_row(tmp_path):
    rows = _rows(_two_pairs(tmp_path))

    asr = [r for r in rows if r["task"] == "asr"]
    st = [r for r in rows if r["task"] == "st"]
    assert len(asr) == 1, "the English side is shared, so it survives once"
    assert sorted(r["tgt_lang"] for r in st) == ["de", "fr"]


def test_asr_text_is_the_source_and_st_text_is_the_target(tmp_path):
    rows = {(r["task"], r["tgt_lang"]): r for r in _rows(_two_pairs(tmp_path))}

    assert rows[("asr", "")]["text"] == "<0.00> hello there<2.00>"
    assert rows[("asr", "")]["text_ctc"] == "hello there"
    # The ST row keeps the English as its CTC target: CTC is always ASR.
    assert rows[("st", "de")]["text"] == "<0.00> hallo da<2.00>"
    assert rows[("st", "de")]["text_ctc"] == "hello there"


def test_utterance_ids_keep_the_two_letter_task_code(tmp_path):
    ids = sorted(r["utt_id"] for r in _rows(_two_pairs(tmp_path)))

    assert ids == [
        "MuST-C_v1.2_t1_000000000_000002000_en_asr",
        "MuST-C_v1.2_t1_000000000_000002000_en_st_de",
        "MuST-C_v1.2_t1_000000000_000002000_en_st_fr",
    ]


def test_the_surviving_asr_copy_is_the_first_language_in_sorted_order(tmp_path):
    """The pairs segment independently, so the copies can differ inside."""
    root = _corpus(
        tmp_path,
        {
            # de splits the window in two, fr keeps it whole: same span id,
            # different text_with_time.
            "de": [
                ("t1.wav", 0.0, 1.0, "first half", "erste"),
                ("t1.wav", 1.0, 1.0, "second half", "zweite"),
            ],
            "fr": [("t1.wav", 0.0, 2.0, "first half second half", "les deux")],
        },
    )
    asr = [r for r in _rows(root) if r["task"] == "asr"]

    (survivor,) = [r for r in asr if r["utt_id"].endswith("000000000_000002000_en_asr")]
    # "de" sorts before "fr", so its two-segment packing wins.
    assert survivor["text"] == "<0.00> first half<1.00><1.00> second half<2.00>"


def test_language_order_does_not_depend_on_the_argument_order(tmp_path):
    root = _two_pairs(tmp_path)
    forward = _rows(root, languages=["de", "fr"])
    backward = _rows(root, languages=["fr", "de"])

    key = lambda rows: sorted((r["utt_id"], r["text"]) for r in rows)  # noqa: E731
    assert key(forward) == key(backward)


def test_whitespace_is_collapsed_and_nothing_else_is_normalised(tmp_path):
    root = _corpus(
        tmp_path,
        {"de": [("t1.wav", 0.0, 2.0, "a  b\tc ", "x --- y — z")]},
    )
    rows = {r["task"]: r for r in _rows(root)}

    assert rows["asr"]["text_ctc"] == "a b c"
    # No Moses punctuation normalisation: the em dash survives.
    assert rows["st"]["text"] == "<0.00> x --- y — z<2.00>"


def test_a_talk_shorter_than_its_segments_is_dropped(tmp_path):
    root = _corpus(
        tmp_path,
        {"de": [("t1.wav", 0.0, 2.0, "kept", "behalten")]},
        talk_seconds=1.0,
    )
    failures = io.StringIO()

    assert _rows(root, failures=failures) == []
    assert "segments run to" in failures.getvalue()


def test_a_talk_at_the_wrong_sample_rate_is_dropped(tmp_path):
    """Durations come from the yaml, so a wrong rate shifts nothing visible.

    It would simply hand the frontend audio at the wrong speed, which is why
    the rate is checked rather than inferred.
    """
    root = _corpus(tmp_path, {"de": [("t1.wav", 0.0, 2.0, "hello", "hallo")]})
    path = root / "en-de" / "data" / "dev" / "wav" / "t1.wav"
    sf.write(str(path), np.zeros(int(60.0 * 8000), dtype="int16"), 8000, "PCM_16")

    failures = io.StringIO()
    assert _rows(root, failures=failures) == []
    assert "sample rate is 8000, expected 16000" in failures.getvalue()


def test_mismatched_line_counts_are_rejected(tmp_path):
    root = _two_pairs(tmp_path)
    txt = root / "en-de" / "data" / "dev" / "txt"
    (txt / "dev.de").write_text("hallo da\nextra line\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="yaml entries"):
        _rows(root)


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        Dataset(split="val", recipe_dir=tmp_path, cache={"cache_dir": str(tmp_path)})
