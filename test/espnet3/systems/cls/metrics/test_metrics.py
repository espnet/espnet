from pathlib import Path

import pytest

from espnet3.systems.cls.metrics.auc import AUC
from espnet3.systems.cls.metrics.macro_f1 import MacroF1
from espnet3.systems.cls.metrics.mean_ap import MAP
from espnet3.systems.cls.metrics.scoring_utils import (
    load_class_labels,
    resolve_classes,
    single_labels,
)
from espnet3.systems.cls.metrics.ua import UA
from espnet3.systems.cls.metrics.wa import WA

sklearn = pytest.importorskip("sklearn")

# Two classes, four utterances. `hyp` is wrong on utt3 only.
REF = ["utt1 joy", "utt2 joy", "utt3 anger", "utt4 anger"]
HYP = ["utt1 joy", "utt2 joy", "utt3 joy", "utt4 anger"]
SCORE = [
    "utt1 0.0 0.9 0.0 0.1 0.0 0.0 0.0",
    "utt2 0.0 0.8 0.0 0.2 0.0 0.0 0.0",
    "utt3 0.0 0.6 0.0 0.4 0.0 0.0 0.0",
    "utt4 0.0 0.3 0.0 0.7 0.0 0.0 0.0",
]
TOKENS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear", "<unk>"]


@pytest.fixture()
def inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name, lines in (("ref", REF), ("hyp", HYP), ("score", SCORE)):
        path = tmp_path / f"{name}.scp"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths[name] = path
    return paths


@pytest.fixture()
def token_list(tmp_path: Path) -> Path:
    path = tmp_path / "token_list"
    path.write_text("\n".join(TOKENS), encoding="utf-8")
    return path


def test_load_class_labels_drops_unk(token_list: Path):
    assert load_class_labels(token_list) == TOKENS[:-1]


def test_load_class_labels_rejects_short_file(tmp_path: Path):
    path = tmp_path / "short"
    path.write_text("only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two entries"):
        load_class_labels(path)


def test_single_labels_rejects_multi_label():
    with pytest.raises(ValueError, match="multi-class only"):
        single_labels(["joy anger"])


def test_resolve_classes_drops_unused(token_list: Path):
    assert resolve_classes(["joy", "anger"], token_list) == ["joy", "anger"]


def test_resolve_classes_rejects_unknown_label(token_list: Path):
    with pytest.raises(ValueError, match="not in the token list"):
        resolve_classes(["excitement"], token_list)


def test_wa(inputs):
    assert WA()(inputs, "test", inputs["ref"].parent) == {"WA": 75.0}


def test_wa_rejects_unaligned_utt_ids(tmp_path: Path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("utt1 joy", encoding="utf-8")
    hyp.write_text("utt2 joy", encoding="utf-8")
    with pytest.raises(AssertionError, match="UID mismatch"):
        WA()({"ref": ref, "hyp": hyp}, "test", tmp_path)


def test_ua_is_macro_recall(inputs, token_list: Path):
    # joy recall 2/2, anger recall 1/2 -> (1.0 + 0.5) / 2
    result = UA(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)
    assert result == {"UA": 75.0}


def test_macro_f1(inputs, token_list: Path):
    # joy F1 = 2*(2/3)*1/(2/3+1) = 0.8, anger F1 = 2*1*0.5/1.5 = 2/3
    result = MacroF1(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)
    assert result == {"MacroF1": 73.33}


def test_map_and_auc_are_perfect_when_scores_rank_correctly(inputs, token_list: Path):
    kwargs = {"token_list": str(token_list)}
    assert MAP(**kwargs)(inputs, "test", inputs["ref"].parent) == {"mAP": 100.0}
    assert AUC(**kwargs)(inputs, "test", inputs["ref"].parent) == {"AUC": 100.0}


def test_map_rejects_wrong_score_width(tmp_path: Path, token_list: Path):
    ref = tmp_path / "ref.scp"
    score = tmp_path / "score.scp"
    ref.write_text("utt1 joy", encoding="utf-8")
    score.write_text("utt1 0.5 0.5", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 7 scores"):
        MAP(token_list=str(token_list))({"ref": ref, "score": score}, "test", tmp_path)
