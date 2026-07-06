import torch.nn as nn

from espnet3.systems.codec.models.gan_lightning_module import GANLightningModule


class _CachingSubModule(nn.Module):
    """Mimics gan_codec/gan_tts modules with cache_generator_outputs + _cache."""

    def __init__(self):
        super().__init__()
        self.cache_generator_outputs = True
        self._cache = object()


class _PlainSubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)


class _ModelWithoutClearCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.plain = _PlainSubModule()
        self.codec = _CachingSubModule()


def test_clear_model_cache_resets_generic_caching_submodule():
    # When skip_discriminator_prob skips the discriminator turn, the stale
    # generator cache must be dropped or the next generator forward reuses
    # a freed autograd graph and crashes on backward().
    module = GANLightningModule.__new__(GANLightningModule)
    nn.Module.__init__(module)
    module.model = _ModelWithoutClearCache()

    assert module.model.codec._cache is not None
    module._clear_model_cache()
    assert module.model.codec._cache is None


def test_clear_model_cache_prefers_explicit_clear_cache_method():
    calls = []

    class _ModelWithClearCache(nn.Module):
        def clear_cache(self):
            calls.append(True)

    module = GANLightningModule.__new__(GANLightningModule)
    nn.Module.__init__(module)
    module.model = _ModelWithClearCache()
    module._clear_model_cache()

    assert calls == [True]
