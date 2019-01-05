#!/usr/bin/env python
import importlib

import h5py
import kaldiio
import numpy as np

from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.io_utils import Preprocessing
from espnet.transform import add_deltas
from espnet.transform import CMVN
from espnet.transform.spectrogram import logmelspectrogram


def make_dummy_json(n_utts=10, ilen_range=(100, 300), olen_range=(10, 300)):
    idim = 83
    odim = 52
    ilens = np.random.randint(ilen_range[0], ilen_range[1], n_utts)
    olens = np.random.randint(olen_range[0], olen_range[1], n_utts)
    dummy_json = {}
    for idx in range(n_utts):
        input = [{
            "shape": [ilens[idx], idim]
        }]
        output = [{
            "shape": [olens[idx], odim]
        }]
        dummy_json["utt_%d" % idx] = {
            "input": input,
            "output": output
        }
    return dummy_json


def test_make_batchset():
    dummy_json = make_dummy_json(128, [128, 512], [16, 128])
    for module in ["espnet.asr.asr_utils", "espnet.tts.tts_utils"]:
        utils = importlib.import_module(module)

        # check w/o adaptive batch size
        batchset = utils.make_batchset(dummy_json, 24, 2**10, 2**10,
                                       min_batch_size=1)
        assert sum([len(batch) >= 1 for batch in batchset]) == len(batchset)
        print([len(batch) for batch in batchset])
        batchset = utils.make_batchset(dummy_json, 24, 2**10, 2**10,
                                       min_batch_size=10)
        assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)
        print([len(batch) for batch in batchset])

        # check w/ adaptive batch size
        batchset = utils.make_batchset(dummy_json, 24, 256, 64,
                                       min_batch_size=10)
        assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)
        print([len(batch) for batch in batchset])
        batchset = utils.make_batchset(dummy_json, 24, 256, 64,
                                       min_batch_size=10)
        assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)


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
    preprocessing = Preprocessing(**kwargs)
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


def test_load_inputs_and_targets_legacy_format(tmpdir):
    # batch = [("F01_050C0101_PED_REAL",
    #          {"input": [{"feat": "some/path.ark:123"}],
    #           "output": [{"tokenid": "1 2 3 4"}],
    ark = str(tmpdir.join('test.ark'))
    scp = str(tmpdir.join('test.scp'))

    desire_xs = []
    desire_ys = []
    with kaldiio.WriteHelper('ark,scp:{},{}'.format(ark, scp)) as f:
        for i in range(10):
            x = np.random.random((100, 100)).astype(np.float32)
            uttid = 'uttid{}'.format(i)
            f[uttid] = x
            desire_xs.append(x)
            desire_ys.append(np.array([1, 2, 3, 4]))

    batch = []
    with open(scp, 'r') as f:
        for line in f:
            uttid, path = line.strip().split()
            batch.append((uttid,
                          {'input': [{'feat': path,
                                      'name': 'input1'}],
                           'output': [{'tokenid': '1 2 3 4',
                                       'name': 'target1'}]}))

    load_inputs_and_targets = LoadInputsAndTargets()
    xs, ys = load_inputs_and_targets(batch)
    for x, xd in zip(xs, desire_xs):
        np.testing.assert_array_equal(x, xd)
    for y, yd in zip(ys, desire_ys):
        np.testing.assert_array_equal(y, yd)


def test_load_inputs_and_targets_new_format(tmpdir):
    # batch = [("F01_050C0101_PED_REAL",
    #           {"input": [{"feat": "some/path.h5",
    #                       "filetype": "hdf5"}],
    #           "output": [{"tokenid": "1 2 3 4"}],

    p = tmpdir.join('test.h5')

    desire_xs = []
    desire_ys = []
    batch = []
    with h5py.File(str(p), 'w') as f:
        # batch: List[Tuple[str, Dict[str, List[Dict[str, Any]]]]]
        for i in range(10):
            x = np.random.random((100, 100)).astype(np.float32)
            uttid = 'uttid{}'.format(i)
            f[uttid] = x
            batch.append((uttid,
                          {'input': [{'feat': str(p) + ':' + uttid,
                                      'filetype': 'hdf5',
                                      'name': 'input1'}],
                           'output': [{'tokenid': '1 2 3 4',
                                       'name': 'target1'}]}))
            desire_xs.append(x)
            desire_ys.append(np.array([1, 2, 3, 4]))

    load_inputs_and_targets = LoadInputsAndTargets()
    xs, ys = load_inputs_and_targets(batch)
    for x, xd in zip(xs, desire_xs):
        np.testing.assert_array_equal(x, xd)
    for y, yd in zip(ys, desire_ys):
        np.testing.assert_array_equal(y, yd)
