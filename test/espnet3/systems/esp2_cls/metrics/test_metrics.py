"""Tests for the ESPnet3 classification metrics.

The five metrics reimplement `egs2/TEMPLATE/asr1/pyscripts/utils/cls_score.py`
as `BaseMetric` classes. Expected values are worked out by hand, and one test
checks the reimplementation still agrees with that script.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from espnet3.systems.esp2_cls.metrics import scoring_utils
from espnet3.systems.esp2_cls.metrics.auc import AUC
from espnet3.systems.esp2_cls.metrics.macro_f1 import MacroF1
from espnet3.systems.esp2_cls.metrics.mean_ap import MAP
from espnet3.systems.esp2_cls.metrics.scoring_utils import (
    build_target_matrix,
    load_class_labels,
    resolve_classes,
    single_labels,
    supported_classes,
)
from espnet3.systems.esp2_cls.metrics.ua import UA
from espnet3.systems.esp2_cls.metrics.wa import WA

sklearn = pytest.importorskip("sklearn")

# ===============================================================
# Test Case Summary
# ===============================================================
#
# scoring_utils
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_load_class_labels_drops_unk            | The trailing entry is not a  |
# |                                             | class.                       |
# | test_load_class_labels_rejects_short_file   | Fewer than two entries is an |
# |                                             | error.                       |
# | test_load_class_labels_requires_trailing_unk | A list without <unk> would  |
# |                                             | lose a real class.           |
# | test_single_labels_rejects_multi_label      | WA/UA/MacroF1 are            |
# |                                             | multi-class only.            |
# | test_resolve_classes_drops_unused           | Classes with no reference    |
# |                                             | example are dropped.         |
# | test_resolve_classes_without_token_list     | Without a token list the     |
# |                          | classes come from the references, sorted.       |
# | test_resolve_classes_rejects_unknown_label  | A label outside the token    |
# |                                             | list is an error.            |
# | test_resolve_classes_rejects_empty_input    | No references is an error.   |
# | test_build_target_matrix_rejects_unknown_label | Same check on the score   |
# |                                             | path.                        |
# | test_supported_classes_rejects_empty_target | A target with no positive    |
# |                                             | anywhere is an error.        |
# | test_require_sklearn_reports_a_missing_install | The import guard raises   |
# |                                             | a usable message.            |
#
# The five metrics
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_wa                                     | Plain accuracy.              |
# | test_wa_rejects_unaligned_utt_ids           | ref and hyp must line up.    |
# | test_wa_rejects_empty_input                 | No utterances is an error.   |
# | test_ua_is_macro_recall                     | Recall averaged over         |
# |                                             | classes.                     |
# | test_macro_f1                               | F1 averaged over classes.    |
# | test_map_is_perfect_when_scores_rank_correctly | Average precision.        |
# | test_auc_is_perfect_when_scores_rank_correctly | Area under the ROC curve. |
# | test_map_skips_classes_without_a_reference  | Classes absent from the      |
# |                          | references are left out of the average.         |
# | test_auc_rejects_a_single_class_reference   | Every utterance in one class |
# |                          | leaves AUC undefined; it must fail, not nan.    |
# | test_auc_skips_a_class_without_a_negative   | A class covering every       |
# |                          | utterance is left out of the average.           |
# | test_map_rejects_wrong_score_width          | The score row must match the |
# |                                             | class count.                 |
#
# Agreement with ESPnet2
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_agrees_with_espnet2_cls_score          | WA / mAP / AUC match         |
# |                          | cls_score.py to two decimal places.             |


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


# ---------------------------------------------------------------
# scoring_utils
# ---------------------------------------------------------------


def test_load_class_labels_drops_unk(token_list: Path):
    assert load_class_labels(token_list) == TOKENS[:-1]


def test_load_class_labels_rejects_short_file(tmp_path: Path):
    path = tmp_path / "short"
    path.write_text("only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two entries"):
        load_class_labels(path)


def test_load_class_labels_requires_trailing_unk(tmp_path: Path):
    path = tmp_path / "token_list"
    path.write_text("\n".join(TOKENS[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must end with <unk>"):
        load_class_labels(path)


def test_single_labels_rejects_multi_label():
    with pytest.raises(ValueError, match="multi-class only"):
        single_labels(["joy anger"])


def test_resolve_classes_drops_unused(token_list: Path):
    assert resolve_classes(["joy", "anger"], token_list) == ["joy", "anger"]


def test_resolve_classes_without_token_list():
    """Ensure the classes come from the references when no list is given."""
    assert resolve_classes(["joy", "anger", "joy"]) == ["anger", "joy"]


def test_resolve_classes_rejects_unknown_label(token_list: Path):
    with pytest.raises(ValueError, match="not in the token list"):
        resolve_classes(["excitement"], token_list)


def test_resolve_classes_rejects_empty_input():
    with pytest.raises(ValueError, match="No reference labels to score"):
        resolve_classes([])


def test_build_target_matrix_rejects_unknown_label():
    with pytest.raises(ValueError, match="not in the token list"):
        build_target_matrix(["excitement"], ["joy", "anger"])


def test_supported_classes_rejects_empty_target():
    with pytest.raises(ValueError, match="No class has a positive reference"):
        supported_classes(np.zeros((2, 3)), ["joy", "anger", "fear"])


def test_require_sklearn_reports_a_missing_install(monkeypatch):
    monkeypatch.setattr(scoring_utils, "sklearn_metrics", None)
    with pytest.raises(RuntimeError, match="scikit-learn is required"):
        scoring_utils.require_sklearn()


# ---------------------------------------------------------------
# The five metrics
# ---------------------------------------------------------------


def test_wa(inputs):
    assert WA()(inputs, "test", inputs["ref"].parent) == {"WA": 75.0}


def test_wa_rejects_unaligned_utt_ids(tmp_path: Path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("utt1 joy", encoding="utf-8")
    hyp.write_text("utt2 joy", encoding="utf-8")
    with pytest.raises(AssertionError, match="UID mismatch"):
        WA()({"ref": ref, "hyp": hyp}, "test", tmp_path)


def test_wa_rejects_empty_input(tmp_path: Path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("", encoding="utf-8")
    hyp.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No utterances to score"):
        WA()({"ref": ref, "hyp": hyp}, "test", tmp_path)


def test_ua_is_macro_recall(inputs, token_list: Path):
    # joy recall 2/2, anger recall 1/2 -> (1.0 + 0.5) / 2
    result = UA(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)
    assert result == {"UA": 75.0}


def test_macro_f1(inputs, token_list: Path):
    # joy F1 = 2*(2/3)*1/(2/3+1) = 0.8, anger F1 = 2*1*0.5/1.5 = 2/3
    result = MacroF1(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)
    assert result == {"MacroF1": 73.33}


def test_map_is_perfect_when_scores_rank_correctly(inputs, token_list: Path):
    result = MAP(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)
    assert result == {"mAP": 100.0}


def test_auc_is_perfect_when_scores_rank_correctly(inputs, token_list: Path):
    result = AUC(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)
    assert result == {"AUC": 100.0}


def test_map_skips_classes_without_a_reference(inputs, token_list: Path, caplog):
    """Ensure the average covers only classes the references mention.

    Five of the seven classes never appear in `REF`, and a class with no
    positive example has no average precision to contribute.
    """
    MAP(token_list=str(token_list))(inputs, "test", inputs["ref"].parent)

    assert "Skipping classes with no reference example" in caplog.text
    assert "neutral" in caplog.text


def test_auc_rejects_a_single_class_reference(tmp_path: Path, token_list: Path):
    """Ensure AUC fails loudly when no class has a negative example.

    scikit-learn returns nan in that case, which would reach metrics.json as
    invalid JSON without anything reporting it.
    """
    ref = tmp_path / "ref.scp"
    score = tmp_path / "score.scp"
    ref.write_text("utt1 joy\nutt2 joy", encoding="utf-8")
    score.write_text(
        "utt1 0.0 0.9 0.0 0.1 0.0 0.0 0.0\nutt2 0.0 0.8 0.0 0.2 0.0 0.0 0.0",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both a positive and a negative"):
        AUC(token_list=str(token_list))({"ref": ref, "score": score}, "test", tmp_path)


def test_auc_skips_a_class_without_a_negative(tmp_path: Path, token_list: Path, caplog):
    """Ensure a class every utterance carries is left out of the average.

    Only multi-label references can reach this: one class covering every
    utterance while another still has both sides.
    """
    ref = tmp_path / "ref.scp"
    score = tmp_path / "score.scp"
    ref.write_text("utt1 joy anger\nutt2 joy", encoding="utf-8")
    score.write_text(
        "utt1 0.0 0.9 0.0 0.8 0.0 0.0 0.0\nutt2 0.0 0.9 0.0 0.2 0.0 0.0 0.0",
        encoding="utf-8",
    )

    result = AUC(token_list=str(token_list))(
        {"ref": ref, "score": score}, "test", tmp_path
    )

    assert result == {"AUC": 100.0}
    assert "Skipping classes with no negative example: ['joy']" in caplog.text


def test_map_rejects_wrong_score_width(tmp_path: Path, token_list: Path):
    ref = tmp_path / "ref.scp"
    score = tmp_path / "score.scp"
    ref.write_text("utt1 joy", encoding="utf-8")
    score.write_text("utt1 0.5 0.5", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 7 scores"):
        MAP(token_list=str(token_list))({"ref": ref, "score": score}, "test", tmp_path)


# ---------------------------------------------------------------
# Agreement with ESPnet2
# ---------------------------------------------------------------

_CLS_SCORE = (
    Path(__file__).resolve().parents[5]
    / "egs2"
    / "TEMPLATE"
    / "asr1"
    / "pyscripts"
    / "utils"
    / "cls_score.py"
)


def _load_cls_score():
    spec = importlib.util.spec_from_file_location("cls_score", _CLS_SCORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agrees_with_espnet2_cls_score(tmp_path: Path):
    """Ensure WA / mAP / AUC still match `cls_score.py` on the same inputs.

    Every class needs at least one reference example: `cls_score.py` averages
    a zero over classes that have none, while these metrics leave them out.
    """
    if not _CLS_SCORE.is_file():
        pytest.skip(f"cls_score.py not found at {_CLS_SCORE}")

    classes = TOKENS[:-1]
    rng = np.random.default_rng(0)
    refs, hyps, scores = [], [], []
    for i, label in enumerate(classes + ["joy", "anger"]):
        utt = f"utt{i}"
        row = rng.random(len(classes))
        # Make the true class win except on the last utterance, so the two
        # implementations are compared on something other than a perfect run.
        if i < len(classes) + 1:
            row[classes.index(label)] = 1.0 + row.max()
        row = row / row.sum()
        refs.append(f"{utt} {label}")
        hyps.append(f"{utt} {classes[int(np.argmax(row))]}")
        scores.append(utt + " " + " ".join(f"{v:.6f}" for v in row))

    paths = {}
    for name, lines in (("ref", refs), ("hyp", hyps), ("score", scores)):
        path = tmp_path / f"{name}.scp"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths[name] = path
    tokens = tmp_path / "token_list"
    tokens.write_text("\n".join(TOKENS), encoding="utf-8")

    espnet2 = _load_cls_score().calc_metrics_from_textfiles(
        str(paths["ref"]), str(paths["hyp"]), str(paths["score"]), str(tokens)
    )
    kwargs = {"token_list": str(tokens)}

    assert WA()(paths, "test", tmp_path)["WA"] == round(espnet2["mean_acc"], 2)
    assert MAP(**kwargs)(paths, "test", tmp_path)["mAP"] == round(espnet2["mAP"], 2)
    assert AUC(**kwargs)(paths, "test", tmp_path)["AUC"] == round(
        espnet2["mean_auc"], 2
    )
