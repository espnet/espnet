from types import SimpleNamespace

from espnet2.gan_svs.espnet_model import ESPnetGANSVSModel


def test_gan_svs_model_clear_cache_delegates_to_svs_clear_cache():
    calls = {}

    class DummySVS:
        def clear_cache(self):
            calls["called"] = True

    model = object.__new__(ESPnetGANSVSModel)
    model.svs = DummySVS()

    model.clear_cache()

    assert calls["called"] is True


def test_gan_svs_model_clear_cache_falls_back_to_internal_cache():
    svs = SimpleNamespace(_cache={"value": 1})
    model = object.__new__(ESPnetGANSVSModel)
    model.svs = svs

    model.clear_cache()

    assert model.svs._cache is None
