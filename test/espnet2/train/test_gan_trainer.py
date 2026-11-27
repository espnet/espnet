import torch

from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.gan_trainer import GANTrainer, GANTrainerOptions
from espnet2.train.reporter import Reporter


class ToyGANModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 1)
        self.discriminator = torch.nn.Linear(1, 1)
        self.forward_generator_calls = []
        self.clear_generator_cache_calls = 0

    def forward(self, x, forward_generator=True):
        self.forward_generator_calls.append(forward_generator)
        if not forward_generator:
            raise AssertionError("The discriminator forward must be skipped")
        loss = self.generator(x).square().mean()
        return {
            "loss": loss,
            "stats": {"loss": loss.detach()},
            "weight": x.new_ones(1),
            "optim_idx": 0,
        }

    def clear_generator_cache(self):
        self.clear_generator_cache_calls += 1


def test_skip_discriminator_clears_generator_cache(tmp_path):
    model = ToyGANModel()
    options = GANTrainerOptions(
        ngpu=0,
        resume=False,
        use_amp=False,
        train_dtype="float32",
        grad_noise=False,
        accum_grad=1,
        grad_clip=5.0,
        grad_clip_type=2.0,
        log_interval=None,
        no_forward_run=False,
        use_matplotlib=False,
        use_tensorboard=False,
        use_wandb=False,
        adapter="lora",
        use_adapter=False,
        save_strategy="all",
        output_dir=tmp_path,
        max_epoch=1,
        seed=0,
        sharded_ddp=False,
        patience=None,
        keep_nbest_models=1,
        nbest_averaging_interval=0,
        early_stopping_criterion=("valid", "loss", "min"),
        best_model_criterion=[("valid", "loss", "min")],
        val_scheduler_criterion=("valid", "loss"),
        unused_parameters=False,
        wandb_model_log_interval=-1,
        create_graph_in_tensorboard=False,
        gradient_as_bucket_view=True,
        ddp_comm_hook=None,
        generator_first=True,
        skip_discriminator_prob=1.0,
    )
    optimizers = [
        torch.optim.SGD(model.generator.parameters(), lr=0.1),
        torch.optim.SGD(model.discriminator.parameters(), lr=0.1),
    ]
    reporter = Reporter()
    reporter.set_epoch(1)

    with reporter.observe("train") as sub_reporter:
        GANTrainer.train_one_epoch(
            model=model,
            iterator=[(["utt"], {"x": torch.ones(1, 2)})],
            optimizers=optimizers,
            schedulers=[None, None],
            scaler=None,
            reporter=sub_reporter,
            summary_writer=None,
            options=options,
            distributed_option=DistributedOption(),
        )

    # The generator runs, but the discriminator is skipped before model.forward().
    assert model.forward_generator_calls == [True]
    # Skipping the discriminator clears the generator output cached for that turn.
    assert model.clear_generator_cache_calls == 1
