import torch

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter
from espnet2.train.trainer import Trainer, TrainerOptions


class ToyModel(AbsESPnetModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.linear(x).pow(2).mean()
        return dict(loss=loss, stats=dict(loss=loss.detach()), weight=x.new_ones(1))

    def collect_feats(self, **kwargs):
        return {}


class ToyIterFactory(AbsIterFactory):
    def build_iter(self, epoch, shuffle=None):
        return [(["utt"], dict(x=torch.ones(1, 2)))]


def _run(output_dir, max_epoch, resume, patience):
    model = ToyModel()
    # With a zero learning rate the loss never improves, so the training is
    # stopped early once the patience is exceeded.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    options = TrainerOptions(
        ngpu=0,
        resume=resume,
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
        output_dir=output_dir,
        max_epoch=max_epoch,
        seed=0,
        sharded_ddp=False,
        patience=patience,
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
    )
    Trainer.run(
        model=model,
        optimizers=[optimizer],
        schedulers=[None],
        train_iter_factory=ToyIterFactory(),
        valid_iter_factory=ToyIterFactory(),
        plot_attention_iter_factory=None,
        trainer_options=options,
        distributed_option=DistributedOption(),
    )
    reporter = Reporter()
    reporter.load_state_dict(torch.load(output_dir / "checkpoint.pth")["reporter"])
    return reporter.get_epoch()


def test_resume_after_early_stopping_does_not_train_more(tmp_path):
    # The loss of epoch 1 is the best, so the training stops after epoch 3.
    assert _run(tmp_path, max_epoch=10, resume=False, patience=1) == 3
    assert _run(tmp_path, max_epoch=10, resume=True, patience=1) == 3


def test_resume_after_early_stopping_with_larger_patience_continues(tmp_path):
    assert _run(tmp_path, max_epoch=10, resume=False, patience=1) == 3
    assert _run(tmp_path, max_epoch=10, resume=True, patience=3) == 5
