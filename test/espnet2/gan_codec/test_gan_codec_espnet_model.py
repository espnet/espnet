from types import SimpleNamespace

from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel


def test_gan_codec_model_clear_cache_delegates_to_codec_clear_cache():
    calls = {}

    class DummyCodec:
        def clear_cache(self):
            calls["called"] = True

    model = object.__new__(ESPnetGANCodecModel)
    model.codec = DummyCodec()

    model.clear_cache()

    assert calls["called"] is True


def test_gan_codec_model_clear_cache_falls_back_to_internal_cache():
    codec = SimpleNamespace(_cache={"value": 1})
    model = object.__new__(ESPnetGANCodecModel)
    model.codec = codec

    model.clear_cache()

    assert model.codec._cache is None
