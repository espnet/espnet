<<<<<<< HEAD
=======
import threading

>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
import kaldiio
import numpy as np

from espnet.transform.add_deltas import add_deltas
from espnet.transform.cmvn import CMVN
<<<<<<< HEAD
from espnet.transform.functional import FuncTrans
from espnet.transform.spectrogram import logmelspectrogram
from espnet.transform.transformation import Transformation
=======
from espnet.transform.spectrogram import logmelspectrogram
from espnet.transform.transformation import global_transform_config
from espnet.transform.transformation import transform_config
from espnet.transform.transformation import Transformation
from espnet.transform.transformation import using_transform_config
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986


def test_preprocessing(tmpdir):
    cmvn_ark = str(tmpdir.join('cmvn.ark'))
    kwargs = {"process": [{"type": "fbank",
                           "n_mels": 80,
                           "fs": 16000,
                           "n_fft": 1024,
                           "n_shift": 512},
                          {"type": "cmvn",
                           "stats": cmvn_ark,
                           "norm_vars": True},
                          {"type": "delta", "window": 2, "order": 2}],
              "mode": "sequential"}

    # Creates cmvn_ark
    samples = np.random.randn(100, 80)
    stats = np.empty((2, 81), dtype=np.float32)
    stats[0, :80] = samples.sum(axis=0)
    stats[1, :80] = (samples ** 2).sum(axis=0)
    stats[0, -1] = 100.
    stats[1, -1] = 0.
    kaldiio.save_mat(cmvn_ark, stats)

    bs = 1
    xs = [np.random.randn(1000).astype(np.float32) for _ in range(bs)]
<<<<<<< HEAD
    preprocessing = Transformation(kwargs)
=======
    preprocessing = Transformation(**kwargs)
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    processed_xs = preprocessing(xs)

    for idx, x in enumerate(xs):
        opt = dict(kwargs['process'][0])
        opt.pop('type')
        x = logmelspectrogram(x, **opt)

        opt = dict(kwargs['process'][1])
        opt.pop('type')
        x = CMVN(**opt)(x)

        opt = dict(kwargs['process'][2])
        opt.pop('type')
        x = add_deltas(x, **opt)

        np.testing.assert_allclose(processed_xs[idx], x)


<<<<<<< HEAD
def test_optional_args():
    kwargs = {"process": [{"type": "channel_selector",
                           "train_channel": 0,
                           "eval_channel": 1,
                           "axis": 0}],
              "mode": "sequential"}
    preprocessing = Transformation(kwargs)
    assert preprocessing(np.array([100, 200]), train=True) == 100
    assert preprocessing(np.array([100, 200]), train=False) == 200


def test_func_trans():
    def foo_bar(x, a=1, b=2):
        '''Foo bar

        :param x: input
        :param int a: default 1
        :param int b: default 2
        '''
        return x + a - b

    class FooBar(FuncTrans):
        _func = foo_bar
        __doc__ = foo_bar.__doc__

    assert FooBar(a=2)(0) == 0
    try:
        FooBar(d=1)
    except TypeError as e:
        raised = True
        assert str(e) == "foo_bar() got an unexpected keyword argument 'd'"
    assert raised
    assert str(FooBar(a=100)) == "FooBar(a=100, b=2)"

    import argparse
    parser = argparse.ArgumentParser()
    FooBar.add_arguments(parser)
    # NOTE: index 0 is help
    assert parser._actions[1].option_strings == ['--foo-bar-a']
    assert parser._actions[1].default == 1
    assert parser._actions[1].type == int
    assert parser._actions[2].option_strings == ['--foo-bar-b']
    assert parser._actions[2].default == 2
    assert parser._actions[2].type == int
=======
def test_using_transform_config():
    # These test could affect globally
    transform_config.reset()

    transform_config['aa'] = False
    with using_transform_config({'aa': True}):
        assert transform_config['aa'] is True
    assert transform_config['aa'] is False
    with using_transform_config({'bb': True}):
        assert transform_config['bb'] is True
    assert 'bb' not in transform_config


def test_transform_config_priority():
    transform_config.reset()

    global_transform_config['aa'] = 3
    assert transform_config['aa'] == 3

    transform_config['aa'] = 2
    assert transform_config['aa'] == 2


def test_transform_config_is_thread_local():
    transform_config.reset()

    def set_transform_config():
        transform_config['aa'] = 2

    def set_global_transform_config():
        global_transform_config['aa'] = 3

    transform_config['aa'] = 1

    t = threading.Thread(target=set_transform_config)
    t.start()
    assert transform_config['aa'] == 1

    t = threading.Thread(target=set_global_transform_config)
    t.start()
    assert global_transform_config['aa'] == 3
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
