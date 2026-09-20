"""Compare checkpoint selection and inference output to ESPnet2."""

import pytest
import torch

from egs3.voices.asr.src.inference import average_checkpoints, build_output


@pytest.mark.parametrize("count", [2, 10])
def test_final_average_matches_native(tmp_path, count):
    """Match native short-run fallback and metric-ranked float summation."""
    from espnet2.main_funcs.average_nbest_models import average_nbest_models
    from espnet2.train.reporter import Reporter

    reporter = Reporter()
    scores = {}
    for epoch in range(count):
        value = ([1e20, -1e20, 3.0] + [0.0] * 10)[epoch]
        score = float({0: 10, 2: 9, 1: 8}.get(epoch, 0))
        state = {"weight": torch.tensor([value]), "count": torch.tensor(epoch + 1)}
        torch.save(state, tmp_path / f"{epoch + 1}epoch.pth")
        path = tmp_path / f"epoch{epoch}_step{epoch + 1}_valid.acc.ckpt"
        torch.save({"state_dict": state}, path)
        scores[str(path)] = torch.tensor(score)
        reporter.stats[epoch + 1] = {"valid": {"acc": score}}
    torch.save(
        {"callbacks": {"scores": {"monitor": "valid/acc", "best_k_models": scores}}},
        tmp_path / f"step{count}.ckpt",
    )
    reporter.set_epoch(count)
    average_nbest_models(tmp_path, reporter, [["valid", "acc", "max"]], 10)
    expected = torch.load(tmp_path / "valid.acc.ave.pth", weights_only=True)
    actual = torch.load(average_checkpoints(tmp_path), weights_only=True)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    with pytest.raises(RuntimeError, match="Expected"):
        average_checkpoints(tmp_path, max_checkpoints=1)
    with pytest.raises(RuntimeError, match="Expected"):
        average_checkpoints(tmp_path / "absent")
    (tmp_path / f"step{count}.ckpt").unlink()
    with pytest.raises(RuntimeError, match="Missing final"):
        average_checkpoints(tmp_path)


def test_output_alignment_and_empty_prediction():
    """Keep every reference, including when a smoke model predicts no text."""
    output = build_output([{"text": "A"}, {"text": "B"}], [[("C",)], [(None,)]], [3, 4])
    assert output == [
        dict(utt_id="3", ref="A", hyp="C"),
        dict(utt_id="4", ref="B", hyp=""),
    ]
    with pytest.raises(ValueError):
        build_output([{"text": "A"}], [], [0])
