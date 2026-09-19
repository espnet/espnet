import json
from pathlib import Path

from espnet3.systems.lid.metrics.accuracy import Accuracy


def test_accuracy_matches_voxlingua_scoring(tmp_path: Path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text(
        "utt1 eng\nutt2 eng\nutt3 fra\nutt4 deu\n",
        encoding="utf-8",
    )
    hyp.write_text(
        "utt1 eng\nutt2 fra\nutt3 fra\nutt4 eng\n",
        encoding="utf-8",
    )

    result = Accuracy()(
        {"ref": ref, "hyp": hyp},
        test_name="dev",
        inference_dir=tmp_path,
    )

    assert result == {
        "Accuracy": 50.0,
        "Precision": 50.0,
        "Recall": 50.0,
        "F1": 50.0,
        "Macro Accuracy": 50.0,
        "Macro Precision": 33.33,
        "Macro Recall": 50.0,
        "Macro F1": 38.89,
    }
    assert (tmp_path / "dev/lid_errors").read_text(encoding="utf-8") == (
        "utt2 eng fra\nutt4 deu eng\n"
    )
    per_language = json.loads((tmp_path / "dev/lid_per_language.json").read_text())
    assert per_language["eng"] == {
        "Count": 2,
        "Correct": 1,
        "Accuracy": 50.0,
        "Precision": 50.0,
        "Recall": 50.0,
        "F1": 50.0,
    }
    assert per_language["fra"]["Precision"] == 50.0
    assert per_language["deu"]["Recall"] == 0.0
    assert json.loads((tmp_path / "dev/lid_error_counts.json").read_text()) == {
        "deu->eng": 1,
        "eng->fra": 1,
    }
