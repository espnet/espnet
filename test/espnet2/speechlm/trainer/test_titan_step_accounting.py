"""Exercise optimizer and batch progress through save intervals and resume."""

from unittest.mock import patch

import pytest
import torch

from espnet2.speechlm.trainer.titan_trainer import TitanTrainer


class TrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def reset_loss_stats(self):
        self._loss_stats = {"loss": torch.tensor(0.0), "count": torch.tensor(0.0)}

    def set_requires_all_reduce(self, enabled):
        pass

    def set_is_last_backward(self, enabled):
        pass

    def forward(self, value, loss_scale, **kwargs):
        self.seen.append(int(value))
        loss = (self.weight - value.float()).square()
        self._loss_stats["loss"] += loss.detach()
        self._loss_stats["count"] += 1
        return loss * loss_scale


class Batches:
    def build_iter(self, global_step, length):
        for value in range(global_step, global_step + length):
            yield {
                "value": torch.tensor(value),
                "loss_masks": torch.ones(1, 1, 1),
                "data_stats": {},
            }


def make_trainer(global_step, max_step):
    trainer = TitanTrainer.__new__(TitanTrainer)
    trainer.model = TrainingModel()
    trainer.train_data_factory = Batches()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    trainer.global_step = global_step
    trainer.max_step = max_step
    trainer.save_interval = 2
    trainer.gradient_accumulation_steps = 2
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.dp_pg = None
    trainer.dp_size = 1
    trainer.max_norm = 1.0
    trainer.global_rank = 0
    trainer.log_interval = 1
    trainer.gc_freq = 1000
    return trainer


@pytest.mark.parametrize(
    "start,end,expected", [(0, 3, list(range(6))), (3, 5, [6, 7, 8, 9])]
)
def test_optimizer_and_batch_progress(start, end, expected):
    trainer = make_trainer(start, end)
    with (
        patch("espnet2.speechlm.trainer.titan_trainer.dist.all_reduce"),
        patch("espnet2.speechlm.trainer.titan_trainer.wandb.log"),
    ):
        while trainer.global_step < trainer.max_step:
            trainer.train()
    assert trainer.global_step == end
    assert trainer.model.seen == expected
    assert torch.isfinite(trainer.model.weight)


def test_exhausted_iterator_does_not_advance_optimizer():
    trainer = make_trainer(0, 2)
    original = trainer.model.weight.detach().clone()
    with (
        patch.object(trainer.train_data_factory, "build_iter", return_value=iter(())),
        patch("espnet2.speechlm.trainer.titan_trainer.dist.all_reduce"),
    ):
        with pytest.raises(RuntimeError, match="iterator ended"):
            trainer.train()
    assert trainer.global_step == 0
    assert trainer.lr_scheduler.last_epoch == 0
    torch.testing.assert_close(trainer.model.weight, original)
