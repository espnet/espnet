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
