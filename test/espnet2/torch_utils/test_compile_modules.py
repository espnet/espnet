import copy
import sys

import pytest
import torch

from espnet2.torch_utils.compile_modules import compile_modules


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
        self.decoder = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def test_compiles_the_named_submodules_only():
    model = Model()
    compile_modules(model, ["encoder"], {"backend": "eager"})
    assert model.encoder._compiled_call_impl is not None
    assert model.decoder._compiled_call_impl is None


def test_dotted_path_and_same_output():
    torch.manual_seed(0)
    model = Model()
    x = torch.randn(3, 4)
    expected = model(x)
    compile_modules(model, ["encoder.0", "decoder"], {"backend": "eager"})
    assert model.encoder[0]._compiled_call_impl is not None
    torch.testing.assert_close(model(x), expected)


def test_state_dict_keys_do_not_change():
    model = Model()
    keys = list(model.state_dict())
    compile_modules(model, ["encoder", "decoder"], {"backend": "eager"})
    assert list(model.state_dict()) == keys
    Model().load_state_dict(model.state_dict())  # loads into an uncompiled model


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        compile_modules(Model(), ["frontend"])


def test_empty_name_raises():
    with pytest.raises(ValueError):
        compile_modules(Model(), [""])


def test_duplicate_or_nested_names_raise():
    with pytest.raises(ValueError):
        compile_modules(Model(), ["encoder", "encoder"])
    with pytest.raises(ValueError):
        compile_modules(Model(), ["encoder", "encoder.0"])


def _train_steps(model, steps=3):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(5, 4, generator=g)
        loss = model(x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.detach()


@pytest.mark.parametrize(
    "conf",
    [
        {"backend": "eager"},
        pytest.param(
            {},  # the default backend, TorchInductor
            marks=[
                pytest.mark.skipif(
                    sys.platform == "win32", reason="Inductor needs a C++ compiler"
                ),
                # Inductor generates and builds the kernels of the forward and
                # backward graphs: longer than the 10 s CI default (it timed out
                # with torch 2.14 on Ubuntu)
                pytest.mark.execution_timeout(180),
            ],
        ),
    ],
)
def test_training_steps_equal_the_uncompiled_model(conf):
    """Forward, backward and optimizer steps through a compiled submodule."""
    torch.manual_seed(0)
    model = Model()
    reference = copy.deepcopy(model)
    compile_modules(model, ["encoder", "decoder"], conf)
    loss = _train_steps(model)
    expected = _train_steps(reference)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    for p, q in zip(model.parameters(), reference.parameters()):
        assert p.grad is not None
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("collect_stats", [False, True])
def test_main_worker_compiles_for_training_only(tmp_path, monkeypatch, collect_stats):
    """AbsTask.main_worker passes the names and the conf, not in collect_stats."""
    import espnet2.tasks.abs_task as abs_task
    from espnet2.tasks.diar import DiarizationTask

    calls = []
    monkeypatch.setattr(
        abs_task,
        "compile_modules",
        lambda model, names, conf: calls.append((type(model).__name__, names, conf)),
    )
    cmd = [
        "--dry_run",
        "true",
        "--output_dir",
        str(tmp_path),
        "--num_spk",
        "1",
        "--torch_compile_modules",
        "frontend",
        "encoder",
        "--torch_compile_conf",
        "backend=eager",
    ]
    if collect_stats:
        cmd += ["--collect_stats", "true"]
    DiarizationTask.main(cmd=cmd)
    if collect_stats:
        assert calls == []
    else:
        assert calls == [
            ("ESPnetDiarizationModel", ["frontend", "encoder"], {"backend": "eager"})
        ]
