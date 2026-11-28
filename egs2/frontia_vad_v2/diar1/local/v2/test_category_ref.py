"""Tests of local/v2/category_ref.py (category-aware test-set references).

python3 -m pytest local/v2/test_category_ref.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import category_ref  # noqa: E402

MAPS = HERE.parents[1] / "conf" / "label_map"


def make_set(tmp_path):
    s = tmp_path / "set"
    s.mkdir()
    (s / "utt2num_samples").write_text("r1 160000\n")
    (s / "wav.scp").write_text("r1 audio/x.omni\n")
    (s / "rttm").write_text(
        "SPEAKER r1 1 1.000 2.000 <NA> <NA> speech <NA> <NA>\n"
        "SPEAKER r1 1 4.000 1.000 <NA> <NA> speech <NA> <NA>\n"
    )
    lab = tmp_path / "labels"
    lab.mkdir()
    (lab / "segments").write_text("u1 r1 1.00 3.00\nu2 r1 4.00 5.00\n")
    (lab / "segment_tags").write_text("u1 speech/utterance <NA>\nu2 speech/blank ba\n")
    (lab / "categories").write_text(
        "r1 1.200 2.500 speech/words <NA> 2\n"
        "r1 6.000 6.500 vocal/laughter laughs 1\n"
        "r1 2.000 2.400 vocal/laughter laugh 1\n"
    )
    return s, lab


def read(path):
    return Path(path).read_text().splitlines()


def run(tmp_path, s, lab, name):
    out = tmp_path / name
    category_ref.main(
        [
            "--set",
            str(s),
            "--labels",
            str(lab),
            "--map",
            str(MAPS / name),
            "--out",
            str(out),
        ]
    )
    return out


def test_v2x_gives_the_set_reference(tmp_path):
    s, lab = make_set(tmp_path)
    out = run(tmp_path, s, lab, "v2x.yaml")
    assert read(out / "rttm") == read(s / "rttm")
    assert read(out / "uem") == ["r1 1 0.000 10.000"]
    assert (out / "wav.scp").is_symlink()


def test_eval_map_leaves_out_blank_and_vocal_outside_speech(tmp_path):
    s, lab = make_set(tmp_path)
    out = run(tmp_path, s, lab, "eval_vocal_ignore.yaml")
    assert read(out / "rttm") == ["SPEAKER r1 1 1.000 2.000 <NA> <NA> speech <NA> <NA>"]
    # the blank utterance (4-5 s) and the laugh outside speech (6-6.5 s) are not
    # scored; the laugh inside the utterance (2-2.4 s) stays speech
    assert read(out / "uem") == [
        "r1 1 0.000 4.000",
        "r1 1 5.000 6.000",
        "r1 1 6.500 10.000",
    ]


def test_word_map_scores_words_and_leaves_out_the_rest_of_utterances(tmp_path):
    s, lab = make_set(tmp_path)
    out = run(tmp_path, s, lab, "eval_words.yaml")
    assert read(out / "rttm") == ["SPEAKER r1 1 1.200 1.300 <NA> <NA> speech <NA> <NA>"]
    assert read(out / "uem") == [
        "r1 1 0.000 1.000",
        "r1 1 1.200 2.500",
        "r1 1 3.000 4.000",
        "r1 1 5.000 6.000",
        "r1 1 6.500 10.000",
    ]


def test_subtract():
    assert category_ref.subtract([(0, 10)], [(2, 3), (5, 6)]) == [
        (0, 2),
        (3, 5),
        (6, 10),
    ]
    assert category_ref.subtract([(0, 1)], [(0, 1)]) == []


def test_alimeeting_marker_only_intervals():
    from prep_eval_corpora import text_labels

    utts, cats = text_labels([(0, 1, "你好"), (1, 2, "[laugh]"), (2, 3, "<sil>")])
    assert [u[2] for u in utts] == [True, False, False]
    assert cats == [(1, 2, "vocal/laughter", ["laugh"], 1)]
