import logging
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from espnet2.train.reporter import aggregate
from espnet2.train.reporter import Average
from espnet2.train.reporter import ReportedValue
from espnet2.train.reporter import Reporter


@pytest.mark.parametrize("weight1,weight2", [(None, None), (19, np.array(9))])
def test_register(weight1, weight2):
    reporter = Reporter()
    reporter.set_epoch(1)
    with reporter.observe(uuid.uuid4().hex) as sub:
        stats1 = {
            "float": 0.6,
            "int": 6,
            "np": np.random.random(),
            "torch": torch.rand(1),
        }
        sub.register(stats1, weight1)
        sub.next()

        stats2 = {
            "float": 0.3,
            "int": 100,
            "np": np.random.random(),
            "torch": torch.rand(1),
        }
        sub.register(stats2, weight2)
        sub.next()
        assert sub.get_epoch() == 1
    with pytest.raises(RuntimeError):
        sub.register({})

    desired = {}
    for k in stats1:
        if stats1[k] is None:
            continue

        if weight1 is None:
            desired[k] = (stats1[k] + stats2[k]) / 2
        else:
            weight1 = float(weight1)
            weight2 = float(weight2)
            desired[k] = float(weight1 * stats1[k] + weight2 * stats2[k])
            desired[k] /= weight1 + weight2

    for k1, k2 in reporter.get_all_keys():
        if k2 in ("time", "total_count", "gpu_max_cached_mem_GB", "gpu_cached_mem_GB"):
            continue
        np.testing.assert_allclose(reporter.get_value(k1, k2), desired[k2])


@pytest.mark.parametrize("mode", ["min", "max", "foo"])
def test_sort_epochs_and_values(mode):
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{"aa": 0.3}, {"aa": 0.5}, {"aa": 0.2}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
    if mode not in ("min", "max"):
        with pytest.raises(ValueError):
            reporter.sort_epochs_and_values(key1, "aa", mode)
        return
    else:
        sort_values = reporter.sort_epochs_and_values(key1, "aa", mode)

    if mode == "min":
        sign = 1
    else:
        sign = -1
    desired = sorted(
        [(e + 1, stats_list[e]["aa"]) for e in range(len(stats_list))],
        key=lambda x: sign * x[1],
    )

    for e in range(len(stats_list)):
        assert sort_values[e] == desired[e]


def test_sort_epochs_and_values_no_key():
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{"aa": 0.3}, {"aa": 0.5}, {"aa": 0.2}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
    with pytest.raises(KeyError):
        reporter.sort_epochs_and_values("foo", "bar", "min")


def test_get_value_not_found():
    reporter = Reporter()
    with pytest.raises(KeyError):
        reporter.get_value("a", "b")


def test_sort_values():
    mode = "min"
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{"aa": 0.3}, {"aa": 0.5}, {"aa": 0.2}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
    sort_values = reporter.sort_values(key1, "aa", mode)

    desired = sorted(
        [stats_list[e]["aa"] for e in range(len(stats_list))],
    )

    for e in range(len(stats_list)):
        assert sort_values[e] == desired[e]


def test_sort_epochs():
    mode = "min"
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{"aa": 0.3}, {"aa": 0.5}, {"aa": 0.2}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
    sort_values = reporter.sort_epochs(key1, "aa", mode)

    desired = sorted(
        [(e + 1, stats_list[e]["aa"]) for e in range(len(stats_list))],
        key=lambda x: x[1],
    )

    for e in range(len(stats_list)):
        assert sort_values[e] == desired[e][0]


def test_best_epoch():
    mode = "min"
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{"aa": 0.3}, {"aa": 0.5}, {"aa": 0.2}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
    best_epoch = reporter.get_best_epoch(key1, "aa", mode)
    assert best_epoch == 3


def test_check_early_stopping():
    mode = "min"
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{"aa": 0.3}, {"aa": 0.2}, {"aa": 0.4}, {"aa": 0.3}]
    patience = 1

    results = []
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
        truefalse = reporter.check_early_stopping(patience, key1, "aa", mode)
        results.append(truefalse)
    assert results == [False, False, False, True]


def test_logging(tmp_path):
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    key2 = uuid.uuid4().hex
    stats_list = [
        {"aa": 0.3, "bb": 3.0},
        {"aa": 0.5, "bb": 3.0},
        {"aa": 0.2, "bb": 3.0},
    ]
    writer = SummaryWriter(tmp_path)
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
            sub.next()
        with reporter.observe(key2) as sub:
            sub.register(stats_list[e])
            sub.next()
            logging.info(sub.log_message())
            logging.info(sub.log_message(-1))
            logging.info(sub.log_message(0, 1))
            sub.tensorboard_add_scalar(writer, -1)
        with pytest.raises(RuntimeError):
            logging.info(sub.log_message())

    logging.info(reporter.log_message())

    with reporter.observe(key1) as sub:
        sub.register({"aa": 0.1, "bb": 0.4})
        sub.next()
        sub.register({"aa": 0.1})
        sub.next()


def test_has_key():
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()
    assert reporter.has(key1, "aa")


def test_get_Keys():
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()
    assert reporter.get_keys() == (key1,)


def test_get_Keys2():
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()
    assert reporter.get_keys2(key1) == ("aa",)


def test_matplotlib_plot(tmp_path: Path):
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()

    reporter.set_epoch(1)
    with reporter.observe(key1) as sub:
        # Skip epoch=2
        sub.register({})
        sub.next()

    reporter.set_epoch(3)
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()

    reporter.matplotlib_plot(tmp_path)
    assert (tmp_path / "aa.png").exists()


def test_tensorboard_add_scalar(tmp_path: Path):
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = "train"
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()

    reporter.set_epoch(1)
    with reporter.observe(key1) as sub:
        # Skip epoch=2
        sub.register({})
        sub.next()

    reporter.set_epoch(3)
    with reporter.observe(key1) as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()

    writer = SummaryWriter(tmp_path)
    reporter.tensorboard_add_scalar(writer)


def test_state_dict():
    reporter = Reporter()
    reporter.set_epoch(1)
    with reporter.observe("train") as sub:
        stats1 = {"aa": 0.6}
        sub.register(stats1)
        sub.next()
    with reporter.observe("eval") as sub:
        stats1 = {"bb": 0.6}
        sub.register(stats1)
        sub.next()
    state = reporter.state_dict()

    reporter2 = Reporter()
    reporter2.load_state_dict(state)
    state2 = reporter2.state_dict()

    assert state == state2


def test_get_epoch():
    reporter = Reporter(2)
    assert reporter.get_epoch() == 2


def test_total_count():
    reporter = Reporter(2)
    assert reporter.get_epoch() == 2
    with reporter.observe("train", 1) as sub:
        sub.register({})
        sub.next()
    with reporter.observe("train", 2) as sub:
        sub.register({})
        sub.next()
        sub.register({})
        sub.next()
        assert sub.get_total_count() == 3


def test_change_epoch():
    reporter = Reporter()
    with pytest.raises(RuntimeError):
        with reporter.observe("train", 1):
            reporter.set_epoch(2)


def test_minus_epoch():
    with pytest.raises(ValueError):
        Reporter(-1)


def test_minus_epoch2():
    reporter = Reporter()
    with pytest.raises(ValueError):
        reporter.set_epoch(-1)
    reporter.start_epoch("aa", 1)
    with pytest.raises(ValueError):
        reporter.start_epoch("aa", -1)


def test_register_array():
    reporter = Reporter()
    with reporter.observe("train", 1) as sub:
        with pytest.raises(ValueError):
            sub.register({"a": np.array([0, 1])})
            sub.next()
        with pytest.raises(ValueError):
            sub.register({"b": 1}, weight=np.array([1, 2]))
            sub.next()


def test_zero_weight():
    reporter = Reporter()
    with reporter.observe("train", 1) as sub:
        sub.register({"a": 1}, weight=0)
        sub.next()


def test_register_nan():
    reporter = Reporter()
    with reporter.observe("train", 1) as sub:
        sub.register({"a": np.nan}, weight=1.0)
        sub.next()


def test_no_register():
    reporter = Reporter()
    with reporter.observe("train", 1):
        pass


def test_mismatch_key2():
    reporter = Reporter()
    with reporter.observe("train", 1) as sub:
        sub.register({"a": 2})
        sub.next()
    with reporter.observe("train", 2) as sub:
        sub.register({"b": 3})
        sub.next()


def test_reserved():
    reporter = Reporter()
    with reporter.observe("train", 1) as sub:
        with pytest.raises(RuntimeError):
            sub.register({"time": 2})
            sub.next()
        with pytest.raises(RuntimeError):
            sub.register({"total_count": 3})
            sub.next()


def test_different_type():
    reporter = Reporter()
    with pytest.raises(ValueError):
        with reporter.observe("train", 1) as sub:
            sub.register({"a": 2}, weight=1)
            sub.next()
            sub.register({"a": 3})
            sub.next()


def test_start_middle_epoch():
    reporter = Reporter()
    with reporter.observe("train", 2) as sub:
        sub.register({"a": 3})
        sub.next()


def test__plot_stats_input_str():
    reporter = Reporter()
    with pytest.raises(TypeError):
        reporter._plot_stats("aaa", "a")


class DummyReportedValue(ReportedValue):
    pass


def test_aggregate():
    vs = [Average(0.1), Average(0.3)]
    assert aggregate(vs) == 0.2
    vs = []
    assert aggregate(vs) is np.nan
    with pytest.raises(NotImplementedError):
        vs = [DummyReportedValue()]
        aggregate(vs)


def test_measure_time():
    reporter = Reporter()
    with reporter.observe("train", 2) as sub:
        with sub.measure_time("foo"):
            pass
        sub.next()


def test_measure_iter_time():
    reporter = Reporter()
    with reporter.observe("train", 2) as sub:
        for _ in sub.measure_iter_time(range(3), "foo"):
            sub.next()
