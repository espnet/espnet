"""Two ranks merged through state/merge_states/load_state equal one run."""

import pytest

from espnet3.api.inference import Field
from espnet3.components.metrics.base_metric import BaseMetric

ROWS = [f"r{i}" * (i + 1) for i in range(10)]
SPLITS = [(5, 5), (9, 1), (10, 0), (0, 10)]


class Counts(BaseMetric):
    inputs = (Field("ref", "text"),)
    outputs = (Field("mean", "number"),)

    def __init__(self):
        super().__init__()
        self.add_state("n", 0, "sum")
        self.add_state("total", 0, "sum")

    def update(self, *, ref):
        self.n += len(ref)
        self.total += sum(len(r) for r in ref)

    def compute(self):
        return {"mean": self.total / self.n}


class Scores(BaseMetric):
    inputs = (Field("ref", "text"),)
    outputs = (Field("mean", "number"), Field("first", "number"))

    def __init__(self):
        super().__init__()
        self.add_state("scores", [], "cat")

    def update(self, *, ref):
        self.scores.extend(len(r) for r in ref)

    def compute(self):
        return {"mean": sum(self.scores) / len(self.scores), "first": self.scores[0]}


def run_ranks(metric_cls, splits):
    """Update one instance per rank, merge their states into a fresh one."""
    ranks, start = [], 0
    for size in splits:
        metric = metric_cls()
        if size:
            metric.update(ref=ROWS[start : start + size])
        start += size
        ranks.append(metric)
    merged = metric_cls()
    merged.load_state(merged.merge_states([m.state() for m in ranks]))
    return ranks, merged


def one_shot(metric_cls):
    metric = metric_cls()
    metric.update(ref=ROWS)
    return metric


@pytest.mark.parametrize("splits", SPLITS)
@pytest.mark.parametrize("metric_cls", [Counts, Scores])
def test_two_ranks_equal_one_run(metric_cls, splits):
    ranks, merged = run_ranks(metric_cls, splits)
    assert merged.compute() == one_shot(metric_cls).compute()
    for size, rank in zip(splits, ranks):
        if size == 0:
            assert rank.state() == metric_cls().state()


@pytest.mark.parametrize("splits", SPLITS)
def test_cat_state_keeps_rank_order(splits):
    _, merged = run_ranks(Scores, splits)
    assert merged.state()["scores"] == [len(r) for r in ROWS]
