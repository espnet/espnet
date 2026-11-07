import torch

from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel


def test_clear_generator_cache():
    model = torch.nn.Module()
    model.codec = torch.nn.Module()
    model.codec._cache = object()

    ESPnetGANCodecModel.clear_generator_cache(model)

    assert model.codec._cache is None
