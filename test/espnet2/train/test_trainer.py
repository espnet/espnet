import copy

import pytest
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
        # loss.detach() shares the storage of loss, as in the ESPnet models
        return dict(loss=loss, stats=dict(loss=loss.detach()), weight=x.new_ones(1))

    def collect_feats(self, **kwargs):
        return {}


class ToyIterFactory(AbsIterFactory):
    def build_iter(self, epoch, shuffle=None):
        return [(["utt"], dict(x=torch.ones(1, 2)))]


def _options(output_dir, max_epoch, resume, patience, accum_grad=1):
    return TrainerOptions(
        ngpu=0,
        resume=resume,
        use_amp=False,
        train_dtype="float32",
        grad_noise=False,
        accum_grad=accum_grad,
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


def _run(output_dir, max_epoch, resume, patience):
    model = ToyModel()
    # With a zero learning rate the loss never improves, so the training is
    # stopped early once the patience is exceeded.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    options = _options(output_dir, max_epoch, resume, patience)
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


def test_train_one_epoch_with_grad_accumulation(tmp_path):
    """accum_grad=2: the same updates as a hand-written loop, every step reported.

    The statistics are registered after backward(), and the reported train loss
    is the mean of the losses: the trainer does not divide the loss in place by
    accum_grad, which would divide loss.detach() in the statistics too.
    """
    torch.manual_seed(0)
    model = ToyModel()
    reference = copy.deepcopy(model)
    batches = [(["utt"], dict(x=torch.full((1, 2), float(i)))) for i in range(1, 5)]

    reporter = Reporter()
    with reporter.observe("train", 1) as sub_reporter:
        Trainer.train_one_epoch(
            model=model,
            iterator=batches,
            optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)],
            schedulers=[None],
            scaler=None,
            reporter=sub_reporter,
            summary_writer=None,
            options=_options(tmp_path, 1, False, 1, accum_grad=2),
            distributed_option=DistributedOption(),
        )

    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    losses = []
    for i, (_, batch) in enumerate(batches, 1):
        loss = reference(**batch)["loss"]
        losses.append(loss.item())
        (loss / 2).backward()
        if i % 2 == 0:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)
    assert reporter.get_value("train", "loss", 1) == pytest.approx(
        sum(losses) / len(losses)
    )
