import torch

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel


class DummyGANModel(AbsGANESPnetModel):
    def forward(self, forward_generator: bool = True, **batch):
        return {
            "loss": torch.tensor(1.0, requires_grad=True),
            "stats": {},
            "weight": torch.tensor(1.0),
            "optim_idx": 0 if forward_generator else 1,
        }

    def collect_feats(self, **batch):
        return {}


def test_abs_gan_espnet_model_clear_cache_is_noop():
    model = DummyGANModel()

    assert model.clear_cache() is None
