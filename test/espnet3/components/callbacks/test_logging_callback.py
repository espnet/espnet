import logging
import re
import time

import torch
import torch.nn as nn

from espnet3.components.callbacks.default_callbacks import (
    MetricsLogger,
)

# | Test Name                         | Description                                                    | # noqa: E501
# |----------------------------------|----------------------------------------------------------------| # noqa: E501
# | test_load_line_basic             | Reads a file with multiple lines like "a\\nb\\nc"               | # noqa: E501
# | test_load_line_with_whitespace   | Strips leading/trailing whitespace from each line              | # noqa: E501
# | test_load_line_empty_file        | Returns an empty list for an empty file                        | # noqa: E501
# | test_load_line_single_line       | Handles file with only one line, no newline                    | # noqa: E501
# | test_load_line_trailing_newline  | Handles file ending with a newline character                   | # noqa: E501


class DummyTrainer:
    def __init__(self, optimizer):
        self.current_epoch = 0
        self.callback_metrics = {}
        self.optimizers = [optimizer]
        self.sanity_checking = False


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


def test_train_batch_metrics_logger_logs_without_training(caplog):
    model = DummyModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer = DummyTrainer(optimizer)
    callback = MetricsLogger(log_every_n_steps=1)

    trainer.callback_metrics = {
        "train/loss": torch.tensor(2.0),
        "train/acc": torch.tensor(0.25),
    }

    batch = ("utt", {"x": torch.zeros(2, 2)})
    with caplog.at_level(logging.INFO):
        callback.on_train_batch_start(trainer, model, batch, batch_idx=0)
        callback.on_before_backward(trainer, model, torch.tensor(1.0))
        callback.on_after_backward(trainer, model)
        callback.on_before_optimizer_step(trainer, model, optimizer)
        callback.on_after_optimizer_step(trainer, model)
        callback.on_train_batch_end(
            trainer, model, outputs=None, batch=batch, batch_idx=0
        )
        callback.on_train_epoch_end(trainer, model)

    text = caplog.text
    assert "1epoch:train:1-1batch" in text
    assert "epoch_summary:1epoch:train:" in text
    assert re.search(r"acc=0\.25\b", text)
    assert re.search(r"loss=2\b", text)
    assert re.search(r"optim0_lr0=0\.02\b", text)


def test_validation_epoch_metrics_logger_logs_epoch_summary(caplog):
    model = DummyModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer = DummyTrainer(optimizer)
    callback = MetricsLogger(log_every_n_steps=1)

    trainer.callback_metrics = {
        "valid/loss": torch.tensor(1.5),
        "valid/acc": torch.tensor(0.75),
        "train/loss": torch.tensor(2.0),
    }

    with caplog.at_level(logging.INFO):
        callback.on_validation_epoch_start(trainer, model)
        time.sleep(0.001)
        callback.on_validation_epoch_end(trainer, model)

    text = caplog.text
    assert "epoch_summary:1epoch:valid:" in text
    assert re.search(r"acc=0\.75\b", text)
    assert re.search(r"loss=1\.5\b", text)
    assert "valid_time=" in text
