from pathlib import Path

import torch

import espnet2.train.gan_trainer as gan_mod
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.gan_trainer import GANTrainer, GANTrainerOptions
from espnet2.train.reporter import SubReporter


class DummyGANModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))
        self.clear_cache_called = 0
        self.forward_calls = []

    def clear_cache(self):
        self.clear_cache_called += 1

    def forward(self, forward_generator=True, **batch):
        self.forward_calls.append(forward_generator)
        loss = self.param * (1.0 if forward_generator else 2.0)
        return {
            "loss": loss,
            "stats": {"loss": loss.detach()},
            "weight": torch.tensor(1.0),
            "optim_idx": 0 if forward_generator else 1,
        }


class DummyDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def make_options(tmp_path, skip_discriminator_prob=1.0):
    return GANTrainerOptions(
        ngpu=0,
        resume=False,
        use_amp=False,
        train_dtype="float32",
        grad_noise=False,
        accum_grad=1,
        grad_clip=0.0,
        grad_clip_type=2.0,
        log_interval=1,
        no_forward_run=False,
        use_matplotlib=False,
        use_tensorboard=False,
        use_wandb=False,
        adapter="",
        use_adapter=False,
        save_strategy="all",
        output_dir=Path(tmp_path),
        max_epoch=1,
        seed=0,
        sharded_ddp=False,
        patience=None,
        keep_nbest_models=1,
        nbest_averaging_interval=0,
        early_stopping_criterion=[],
        best_model_criterion=[],
        val_scheduler_criterion=[],
        unused_parameters=False,
        wandb_model_log_interval=0,
        create_graph_in_tensorboard=False,
        gradient_as_bucket_view=False,
        ddp_comm_hook=None,
        generator_first=False,
        skip_discriminator_prob=skip_discriminator_prob,
    )


def test_gan_trainer_skips_discriminator_and_clears_cache(tmp_path, monkeypatch):
    model = DummyGANModule()
    optimizer_g = torch.optim.SGD([model.param], lr=0.1)
    optimizer_d = torch.optim.SGD([model.param], lr=0.1)
    iterator = [(["utt"], {"x": torch.tensor([1.0])})]

    monkeypatch.setattr(
        gan_mod.torch, "rand", lambda *args, **kwargs: torch.tensor([0.0])
    )

    all_invalid = GANTrainer.train_one_epoch(
        model=model,
        iterator=iterator,
        optimizers=[optimizer_g, optimizer_d],
        schedulers=[None, None],
        scaler=None,
        reporter=SubReporter("train", 1, 0),
        summary_writer=None,
        options=make_options(tmp_path),
        distributed_option=DistributedOption(),
    )

    assert all_invalid is False
    assert model.clear_cache_called == 1
    assert model.forward_calls == [True]


def test_gan_trainer_skips_discriminator_and_clears_cache_for_ddp(
    tmp_path, monkeypatch
):
    wrapped = DummyGANModule()
    model = DummyDDP(wrapped)
    optimizer_g = torch.optim.SGD([wrapped.param], lr=0.1)
    optimizer_d = torch.optim.SGD([wrapped.param], lr=0.1)
    iterator = [(["utt"], {"x": torch.tensor([1.0])})]

    monkeypatch.setattr(gan_mod, "DDP", DummyDDP)
    monkeypatch.setattr(
        gan_mod.torch, "rand", lambda *args, **kwargs: torch.tensor([0.0])
    )

    GANTrainer.train_one_epoch(
        model=model,
        iterator=iterator,
        optimizers=[optimizer_g, optimizer_d],
        schedulers=[None, None],
        scaler=None,
        reporter=SubReporter("train", 1, 0),
        summary_writer=None,
        options=make_options(tmp_path),
        distributed_option=DistributedOption(),
    )

    assert wrapped.clear_cache_called == 1
    assert wrapped.forward_calls == [True]
