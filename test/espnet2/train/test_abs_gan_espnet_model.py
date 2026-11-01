import pytest

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel


def test_clear_generator_cache_is_not_implemented_by_default():
    with pytest.raises(
        NotImplementedError, match="does not support skipping the discriminator"
    ):
        AbsGANESPnetModel.clear_generator_cache(object())
