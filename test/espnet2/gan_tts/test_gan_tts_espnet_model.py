from types import SimpleNamespace

from espnet2.gan_tts.espnet_model import ESPnetGANTTSModel


def test_gan_tts_model_clear_cache_delegates_to_tts_clear_cache():
    calls = {}

    class DummyTTS:
        def clear_cache(self):
            calls["called"] = True

    model = object.__new__(ESPnetGANTTSModel)
    model.tts = DummyTTS()

    model.clear_cache()

    assert calls["called"] is True


def test_gan_tts_model_clear_cache_falls_back_to_internal_cache():
    tts = SimpleNamespace(_cache={"value": 1})
    model = object.__new__(ESPnetGANTTSModel)
    model.tts = tts

    model.clear_cache()

    assert model.tts._cache is None
