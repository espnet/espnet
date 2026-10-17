import pytest

from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric


class Counter(BaseMetric):
    inputs = (Field("ref", "text"),)
    outputs = (Field("n", "number"),)

    def __init__(self):
        super().__init__()
        self.add_state("n", 0, "sum")
        self.add_state("scores", [], "cat")

    def update(self, *, ref):
        self.n += len(ref)
        self.scores.extend(len(r) for r in ref)

    def compute(self):
        return {"n": self.n}


def test_add_state_sets_the_default():
    m = Counter()
    assert m.n == 0 and m.scores == []


def test_add_state_rejects_a_duplicate_name():
    with pytest.raises(ValueError, match="already defined"):
        Counter().add_state("n", 0, "sum")


@pytest.mark.parametrize(
    "default, reduce", [(0, "max"), ([], "sum"), (True, "sum"), (0, "cat")]
)
def test_add_state_rejects_a_bad_reduce_or_default(default, reduce):
    with pytest.raises(ValueError):
        Counter().add_state("x", default, reduce)


def test_reset_restores_the_defaults():
    m = Counter()
    m.update(ref=["ab"])
    m.reset()
    assert m.state() == {"n": 0, "scores": []}


def test_state_returns_copies():
    m = Counter()
    s = m.state()
    s["scores"].append(1)
    assert m.scores == []


def test_merge_states_sums_and_concatenates():
    m = Counter()
    merged = m.merge_states([{"n": 2, "scores": [1]}, {"n": 3, "scores": [2, 3]}])
    assert merged == {"n": 5, "scores": [1, 2, 3]}
    assert m.n == 0


def test_merge_states_of_nothing_is_the_defaults():
    assert Counter().merge_states([]) == {"n": 0, "scores": []}


def test_merge_states_rejects_other_names():
    with pytest.raises(ValueError):
        Counter().merge_states([{"n": 1}])


def test_load_state_then_compute():
    m = Counter()
    m.load_state({"n": 7, "scores": []})
    assert m.compute() == {"n": 7}


def test_add_state_works_without_super_init():
    class NoInit(Counter):
        def __init__(self):
            self.add_state("n", 0, "sum")

    assert NoInit().state() == {"n": 0}


def test_a_metric_without_update_or_compute_cannot_be_built():
    class NoCompute(BaseMetric):
        inputs = (Field("ref", "text"),)
        outputs = (Field("n", "number"),)

        def update(self, *, ref):
            pass

    with pytest.raises(TypeError):
        NoCompute()


def test_a_metric_without_update_cannot_be_built():
    class NoUpdate(BaseMetric):
        inputs = (Field("ref", "text"),)
        outputs = (Field("n", "number"),)

        def compute(self):
            return {"n": 0}

    with pytest.raises(TypeError):
        NoUpdate()


class Scores(BaseMetric):
    inputs = (Field("ref", "text"),)
    outputs = (Field("mean", "number"),)

    def __init__(self):
        super().__init__()
        self.add_state("scores", [], "cat")

    def update(self, *, ref):
        self.scores.extend(len(r) for r in ref)

    def compute(self):
        return {"mean": sum(self.scores) / len(self.scores)}


def test_a_cat_state_merged_across_ranks_scores_like_one_run():
    rows = ["a", "bb", "ccc", "dddd"]
    rank0, rank1, merged = Scores(), Scores(), Scores()
    rank0.update(ref=rows[:2])
    rank1.update(ref=rows[2:])
    merged.load_state(merged.merge_states([rank0.state(), rank1.state()]))
    whole = Scores()
    whole.update(ref=rows)
    assert merged.compute() == whole.compute() == {"mean": 2.5}


def test_callback_round_trip_equals_one_shot():
    rows = ["a", "bb", "ccc", "dddd"]
    ranks = [Counter(), Counter()]
    for rank, metric in enumerate(ranks):
        metric.reset()
        for batch in (rows[rank * 2 : rank * 2 + 1], rows[rank * 2 + 1 : rank * 2 + 2]):
            metric.update(ref=batch)
    states = [m.state() for m in ranks]
    merged = Counter()
    merged.load_state(merged.merge_states(states))
    whole = Counter()
    whole.update(ref=rows)
    assert merged.compute() == whole.compute()
    for state in states:
        assert all(isinstance(v, (int, float, list)) for v in state.values())
