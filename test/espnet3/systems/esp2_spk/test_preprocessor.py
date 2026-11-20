import numpy as np
import pytest

from espnet3.systems.esp2_spk.src.preprocessor import (
    SpkPreprocessor,
    speed_perturbed_speaker,
)


@pytest.fixture
def spk2utt(tmp_path):
    path = tmp_path / "spk2utt"
    path.write_text("id10001 u1 u2\nid10002 u3\n", encoding="utf-8")
    return str(path)


def _preprocessor(spk2utt, **kwargs):
    return SpkPreprocessor(
        train=True,
        target_duration=0.1,
        spk2utt=spk2utt,
        rir_apply_prob=0.0,
        noise_apply_prob=0.0,
        **kwargs,
    )


def test_speed_perturbed_speaker_follows_espnet2():
    assert speed_perturbed_speaker("id10001", 0.9) == "sp0.9-id10001"
    assert speed_perturbed_speaker("id10001", 1) == "id10001"


def test_label_space_is_unchanged_without_perturbation(spk2utt):
    preprocessor = _preprocessor(spk2utt)
    assert preprocessor.spk2label == {"id10001": 0, "id10002": 1}


def test_perturbed_speakers_extend_the_label_space(spk2utt):
    preprocessor = _preprocessor(spk2utt, speed_perturb_factors=[0.9, 1.0, 1.1])

    assert preprocessor.nspk == 6
    # The original speakers keep their labels; each factor adds a block.
    assert preprocessor.spk2label == {
        "id10001": 0,
        "id10002": 1,
        "sp0.9-id10001": 2,
        "sp0.9-id10002": 3,
        "sp1.1-id10001": 4,
        "sp1.1-id10002": 5,
    }
    out = preprocessor(
        "u", {"speech": np.zeros(3200, np.float32), "spk_labels": "sp1.1-id10002"}
    )
    assert int(np.asarray(out["spk_labels"]).reshape(-1)[0]) == 5
