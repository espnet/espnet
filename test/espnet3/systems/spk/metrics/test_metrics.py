from pathlib import Path

import pytest

from espnet3.systems.spk.metrics.eer import EER
from espnet3.systems.spk.metrics.min_dcf import MinDCF


def _build_inputs(tmp_path: Path, scores, labels) -> dict[str, Path]:
    score_path = tmp_path / "score.scp"
    label_path = tmp_path / "label.scp"
    score_path.write_text(
        "\n".join(f"{i} {value}" for i, value in enumerate(scores)), encoding="utf-8"
    )
    label_path.write_text(
        "\n".join(f"{i} {value}" for i, value in enumerate(labels)), encoding="utf-8"
    )
    return {"score": score_path, "label": label_path}


def test_eer_is_zero_for_separable_scores(tmp_path: Path):
    data = _build_inputs(tmp_path, [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])

    assert EER()(data, "vox1_o", tmp_path) == {"EER": 0.0}


def test_eer_is_fifty_percent_for_inverted_scores(tmp_path: Path):
    data = _build_inputs(tmp_path, [0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])

    assert EER()(data, "vox1_o", tmp_path) == {"EER": 100.0}


def test_eer_writes_score_distribution(tmp_path: Path):
    data = _build_inputs(tmp_path, [1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])

    EER()(data, "vox1_o", tmp_path)

    written = (tmp_path / "vox1_o" / "score_distribution").read_text()
    assert "n_trials 4" in written
    assert "target 1.0000 +- 0.0000" in written
    assert "nontarget 0.0000 +- 0.0000" in written


def test_min_dcf_is_zero_for_separable_scores(tmp_path: Path):
    data = _build_inputs(tmp_path, [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])

    assert MinDCF()(data, "vox1_o", tmp_path) == {"minDCF": 0.0}


def test_min_dcf_follows_the_configured_operating_point(tmp_path: Path):
    data = _build_inputs(tmp_path, [0.5, 0.4, 0.6, 0.3], [1, 1, 0, 0])

    lenient = MinDCF(p_target=0.5)(data, "vox1_o", tmp_path)["minDCF"]
    strict = MinDCF(p_target=0.01)(data, "vox1_o", tmp_path)["minDCF"]

    assert strict > lenient


def test_metrics_reject_unaligned_trial_ids(tmp_path: Path):
    data = _build_inputs(tmp_path, [0.9], [1])
    (tmp_path / "label.scp").write_text("7 1", encoding="utf-8")

    with pytest.raises(AssertionError):
        EER()(data, "vox1_o", tmp_path)
