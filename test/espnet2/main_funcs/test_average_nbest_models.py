import pytest
import torch

from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.train.reporter import Reporter


@pytest.fixture
def reporter():
    _reporter = Reporter()
    _reporter.set_epoch(1)
    with _reporter.observe("valid") as sub:
        sub.register({"acc": 0.4})
        sub.next()

    _reporter.set_epoch(2)
    with _reporter.observe("valid") as sub:
        sub.register({"acc": 0.5})
        sub.next()

    _reporter.set_epoch(3)
    with _reporter.observe("valid") as sub:
        sub.register({"acc": 0.6})
        sub.next()

    return _reporter


@pytest.fixture
def output_dir(tmp_path):
    _output_dir = tmp_path / "out"
    _output_dir.mkdir()
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 3),
        torch.nn.BatchNorm2d(1),
        torch.nn.Linear(1, 1),
        torch.nn.LSTM(1, 1),
        torch.nn.LayerNorm(1),
    )
    torch.save(model.state_dict(), _output_dir / "1epoch.pth")
    torch.save(model.state_dict(), _output_dir / "2epoch.pth")
    torch.save(model.state_dict(), _output_dir / "3epoch.pth")
    return _output_dir


@pytest.mark.parametrize("nbest", [0, 1, 2, 3, 4, [1, 2, 3, 5], []])
def test_average_nbest_models(reporter, output_dir, nbest):
    # Repeat twice to check the case of existing files.
    for _ in range(2):
        average_nbest_models(
            reporter=reporter,
            output_dir=output_dir,
            best_model_criterion=[("valid", "acc", "max")],
            nbest=nbest,
        )


@pytest.mark.parametrize("nbest", [0, 1, 2, 3, 4, [1, 2, 3, 5], []])
def test_average_nbest_models_0epoch_reporter(output_dir, nbest):
    # Repeat twice to check the case of existing files.
    for _ in range(2):
        average_nbest_models(
            reporter=Reporter(),
            output_dir=output_dir,
            best_model_criterion=[("valid", "acc", "max")],
            nbest=nbest,
        )


def test_average_nbest_models_reads_each_checkpoint_as_it_was_saved(tmp_path):
    # A checkpoint is loaded once and kept for the other criteria and nbest
    # values, so an average must not be written into it.
    reporter = Reporter()
    for e in (1, 2, 3):
        reporter.set_epoch(e)
        with reporter.observe("valid") as sub:
            # epoch 1 is the best by both criteria, epoch 3 the worst
            sub.register({"loss": float(e), "acc": 1.0 - 0.1 * e})
            sub.next()
        torch.save({"w": torch.tensor([float(e)])}, tmp_path / f"{e}epoch.pth")

    average_nbest_models(
        reporter=reporter,
        output_dir=tmp_path,
        best_model_criterion=[("valid", "loss", "min"), ("valid", "acc", "max")],
        nbest=[2, 3],
    )

    expected = {
        "valid.loss.ave_2best.pth": 1.5,
        "valid.loss.ave_3best.pth": 2.0,
        "valid.acc.ave_2best.pth": 1.5,
        "valid.acc.ave_3best.pth": 2.0,
    }
    for name, value in expected.items():
        assert torch.load(tmp_path / name)["w"].item() == pytest.approx(value), name
