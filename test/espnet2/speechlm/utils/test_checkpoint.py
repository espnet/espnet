"""Resume selects only completed checkpoints, ordered by numeric step."""

from espnet2.speechlm.utils.checkpoint import latest_checkpoint


def test_latest_complete_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for name in ("step_2", "step_10", "step_11", "step_old"):
        path = tmp_path / "checkpoints" / name
        path.mkdir(parents=True)
        if name != "step_11":
            (path / ".metadata").touch()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoints" / "step_10"
