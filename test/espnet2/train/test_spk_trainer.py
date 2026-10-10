from unittest import mock

import numpy as np
import torch

from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter
from espnet2.train.spk_trainer import SpkTrainer
from espnet2.train.trainer import TrainerOptions


class _Embedder(torch.nn.Module):
    def forward(self, speech, spk_labels=None, extract_embd=False, task_tokens=None):
        return speech


def test_validate_one_epoch_score_statistics():
    torch.manual_seed(0)
    emb = {u: torch.randn(1, 4) + 1.0 for u in "abcdef"}
    trials = [("a", "b", 1), ("c", "d", 0), ("e", "f", 0), ("a", "d", 0), ("b", "e", 1)]
    batches = [
        (
            [f"{x}*{y}"],
            {
                "speech": emb[x][None],
                "speech2": emb[y][None],
                "spk_labels": torch.tensor([[t]]),
            },
        )
        for x, y, t in trials
    ]
    reporter = Reporter(epoch=1)
    options = mock.MagicMock(spec=TrainerOptions, ngpu=0)
    dist = mock.MagicMock(spec=DistributedOption, distributed=False)

    with reporter.observe("valid") as sub:
        SpkTrainer.validate_one_epoch(_Embedder(), batches, sub, options, dist)

    stats = {
        key: reporter.get_value("valid", key)
        for key in ("trg_mean", "trg_std", "nontrg_mean", "nontrg_std")
    }
    # The trainer scores a trial by the negative distance of unit-length embeddings.
    unit = {u: torch.nn.functional.normalize(v, dim=1) for u, v in emb.items()}

    def score(x, y):
        return -np.linalg.norm(unit[x].numpy() - unit[y].numpy())

    trg = [score(x, y) for x, y, t in trials if t == 1]
    nontrg = [score(x, y) for x, y, t in trials if t == 0]
    np.testing.assert_allclose(stats["trg_mean"], np.mean(trg), rtol=1e-5)
    np.testing.assert_allclose(stats["trg_std"], np.std(trg), rtol=1e-5)
    np.testing.assert_allclose(stats["nontrg_mean"], np.mean(nontrg), rtol=1e-5)
    np.testing.assert_allclose(stats["nontrg_std"], np.std(nontrg), rtol=1e-5)
