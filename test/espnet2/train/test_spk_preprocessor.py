import numpy as np
import pytest

from espnet2.train.preprocessor import SpkPreprocessor


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("task_token", [None, "0", "3"])
def test_task_tokens_are_integer_arrays(tmp_path, train, task_token):
    labels = tmp_path / "spk2utt"
    labels.write_text("speaker utterance\n")
    preprocessor = SpkPreprocessor(
        train=train,
        target_duration=0.05,
        spk2utt=str(labels),
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
    )
    data = {
        "speech": np.ones(1600, dtype=np.float32),
        "spk_labels": "speaker" if train else "0",
    }
    if not train:
        data["speech2"] = np.ones(1600, dtype=np.float32)
    if task_token is not None:
        data["task_tokens"] = task_token
    output = preprocessor("utterance", data)
    assert output["spk_labels"].item() == 0
    if task_token is None:
        assert "task_tokens" not in output
    else:
        assert output["task_tokens"].shape == (1,)
        assert output["task_tokens"].dtype.kind == "i"
        assert output["task_tokens"].item() == int(task_token)
