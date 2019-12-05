import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.train.reporter import aggregate
from espnet2.train.reporter import Average
from espnet2.train.reporter import ReportedValue
from espnet2.train.reporter import Reporter


@pytest.mark.parametrize(
    'weight1,weight2', [(None, None), (19, np.array(9))])
def test_register(weight1, weight2):
    reporter = Reporter()
    reporter.set_epoch(1)
    with reporter.observe(uuid.uuid4().hex) as sub:
        stats1 = {'float': 0.6,
                  'int': 6,
                  'np': np.random.random(),
                  'torch': torch.rand(1),
                  'none': None}
        sub.register(stats1, weight1)
        stats2 = {'float': 0.3,
                  'int': 100,
                  'np': np.random.random(),
                  'torch': torch.rand(1),
                  'none': None,
                  }
        sub.register(stats2, weight2)
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
            desired[k] /= (weight1 + weight2)

    for k1, k2 in reporter.get_all_keys():
        if k2 in ('time', 'total_count'):
            continue
        np.testing.assert_allclose(reporter.get_value(k1, k2),
                                   desired[k2])


@pytest.mark.parametrize('mode', ['min', 'max', 'foo'])
def test_sort_values(mode):
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    stats_list = [{'aa': 0.3}, {'aa': 0.5}, {'aa': 0.2}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
    if mode not in ('min', 'max'):
        with pytest.raises(ValueError):
            reporter.sort_epochs_and_values(key1, 'aa', mode)
        return
    else:
        sort_values = reporter.sort_epochs_and_values(key1, 'aa', mode)

    if mode == 'min':
        sign = 1
    else:
        sign = -1
    desired = sorted([(e + 1, stats_list[e]['aa'])
                      for e in range(len(stats_list))],
                     key=lambda x: sign * x[1])

    for e in range(len(stats_list)):
        assert sort_values[e] == desired[e]


def test_logging():
    reporter = Reporter()
    key1 = uuid.uuid4().hex
    key2 = uuid.uuid4().hex
    stats_list = [{'aa': 0.3, 'bb': 3.},
                  {'aa': 0.5, 'bb': 3.},
                  {'aa': 0.2, 'bb': 3.}]
    for e in range(len(stats_list)):
        reporter.set_epoch(e + 1)
        with reporter.observe(key1) as sub:
            sub.register(stats_list[e])
        with reporter.observe(key2) as sub:
            sub.register(stats_list[e])
            sub.logging()
        with pytest.raises(RuntimeError):
            sub.logging()

    reporter.logging()


def test_has_key():
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {'aa': 0.6}
        sub.register(stats1)
    assert reporter.has(key1, 'aa')


def test_get_Keys():
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {'aa': 0.6}
        sub.register(stats1)
    assert reporter.get_keys() == (key1,)


def test_get_Keys2():
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {'aa': 0.6}
        sub.register(stats1)
    assert reporter.get_keys2(key1) == ('aa',)


def test_save_stats_plot(tmp_path: Path):
    reporter = Reporter()
    reporter.set_epoch(1)
    key1 = uuid.uuid4().hex
    with reporter.observe(key1) as sub:
        stats1 = {'aa': 0.6}
        sub.register(stats1)

    reporter.set_epoch(1)
    with reporter.observe(key1) as sub:
        # Skip epoch=2
        sub.register({})

    reporter.set_epoch(3)
    with reporter.observe(key1) as sub:
        stats1 = {'aa': 0.6}
        sub.register(stats1)

    reporter.save_stats_plot(tmp_path)
    assert (tmp_path / 'aa.png').exists()


def test_state_dict():
    reporter = Reporter()
    reporter.set_epoch(1)
    with reporter.observe('train') as sub:
        stats1 = {'aa': 0.6}
        sub.register(stats1)
    with reporter.observe('eval') as sub:
        stats1 = {'bb': 0.6}
        sub.register(stats1)
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
    with reporter.observe('train', 1) as sub:
        sub.register({})
    with reporter.observe('train', 2) as sub:
        sub.register({})
        sub.register({})
        assert sub.get_total_count() == 3


def test_change_epoch():
    reporter = Reporter()
    with pytest.raises(RuntimeError):
        with reporter.observe('train', 1):
            reporter.set_epoch(2)


def test_minus_epoch():
    with pytest.raises(ValueError):
        Reporter(-1)


def test_minus_epoch2():
    reporter = Reporter()
    with pytest.raises(ValueError):
        reporter.set_epoch(-1)
    reporter.start_epoch('aa', 1)
    with pytest.raises(ValueError):
        reporter.start_epoch('aa', -1)


def test_register_array():
    reporter = Reporter()
    with reporter.observe('train', 1) as sub:
        with pytest.raises(ValueError):
            sub.register({'a': np.array([0, 1])})
        with pytest.raises(ValueError):
            sub.register({'a': 1}, weight=np.array([1, 2]))


def test_zero_weight():
    reporter = Reporter()
    with reporter.observe('train', 1) as sub:
        sub.register({'a': 1}, weight=0)


def test_register_nan():
    reporter = Reporter()
    with reporter.observe('train', 1) as sub:
        sub.register({'a': np.nan}, weight=1.0)


def test_no_register():
    reporter = Reporter()
    with reporter.observe('train', 1):
        pass


def test_mismatch_key2():
    reporter = Reporter()
    with reporter.observe('train', 1) as sub:
        sub.register({'a': 2})
    with reporter.observe('train', 2) as sub:
        sub.register({'b': 3})


def test_reserved():
    reporter = Reporter()
    with reporter.observe('train', 1) as sub:
        with pytest.raises(RuntimeError):
            sub.register({'time': 2})
        with pytest.raises(RuntimeError):
            sub.register({'total_count': 3})


def test_different_type():
    reporter = Reporter()
    with pytest.raises(ValueError):
        with reporter.observe('train', 1) as sub:
            sub.register({'a': 2}, weight=1)
            sub.register({'a': 3})


def test_start_middle_epoch():
    reporter = Reporter()
    with reporter.observe('train', 2) as sub:
        sub.register({'a': 3})


def test__plot_stats_input_str():
    reporter = Reporter()
    with pytest.raises(TypeError):
        reporter._plot_stats('aaa', 'a')


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
