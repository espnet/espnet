#!/usr/bin/env python
import importlib
import subprocess

import h5py
import numpy as np
import kaldi_io_py
import pytest

from espnet.asr.asr_utils import LoadInputsAndTargets


def make_dummy_json(n_utts=10, ilen_range=[100, 300], olen_range=[10, 300]):
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
        batchset = utils.make_batchset(dummy_json, 24, 2**10, 2**10, min_batch_size=1)
        assert sum([len(batch) >= 1 for batch in batchset]) == len(batchset)
        print([len(batch) for batch in batchset])
        batchset = utils.make_batchset(dummy_json, 24, 2**10, 2**10, min_batch_size=10)
        assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)
        print([len(batch) for batch in batchset])

        # check w/ adaptive batch size
        batchset = utils.make_batchset(dummy_json, 24, 256, 64, min_batch_size=10)
        assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)
        print([len(batch) for batch in batchset])
        batchset = utils.make_batchset(dummy_json, 24, 256, 64, min_batch_size=10)
        assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)


def test_load_inputs_and_targets_legacy_format(tmpdir):
    # (shutil.which doesn't exist in Python2)
    if subprocess.run(['which', 'copy-feats']).returncode != 0:
        pytest.skip('You don\'t have copy-feats')
    # batch = [("F01_050C0101_PED_REAL",
    #          {"input": [{"feat": "some/path.ark:123"}],
    #           "output": [{"tokenid": "1 2 3 4"}],
    ark = str(tmpdir.join('test.ark'))
    scp = str(tmpdir.join('test.scp'))
    wspec = 'ark:| copy-feats ark:- ark,scp:{},{}'.format(ark, scp)

    desire_xs = []
    desire_ys = []
    with kaldi_io_py.open_or_fd(wspec, 'wb') as f:
        for i in range(10):
            x = np.random.random((100, 100)).astype(np.float32)
            uttid = 'uttid{}'.format(i)
            kaldi_io_py.write_mat(f, x, key=uttid)
            desire_xs.append(x)
            desire_ys.append(np.array([1, 2, 3, 4]))

    batch = []
    with open(scp, 'r') as f:
        for line in f:
            uttid, path = line.strip().split()
            batch.append((uttid,
                          {'input': [{'feat': path}],
                           'output': [{'tokenid': '1 2 3 4'}]}))

    load_inputs_and_targets = LoadInputsAndTargets()
    xs, ys = load_inputs_and_targets(batch)
    for x, xd in zip(xs, desire_xs):
        np.testing.assert_array_equal(x, xd)
    for y, yd in zip(ys, desire_ys):
        np.testing.assert_array_equal(y, yd)


def test_load_inputs_and_targets_new_format(tmpdir):
    # batch = [("F01_050C0101_PED_REAL",
    #           {"input": [{"path": "some/path.h5",
    #                       "type": "hdf5"}],
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
                          {'input': [{'path': str(p),
                                      'type': 'hdf5'}],
                           'output': [{'tokenid': '1 2 3 4'}]}))
            desire_xs.append(x)
            desire_ys.append(np.array([1, 2, 3, 4]))

    load_inputs_and_targets = LoadInputsAndTargets()
    xs, ys = load_inputs_and_targets (batch)
    for x, xd in zip(xs, desire_xs):
        np.testing.assert_array_equal(x, xd)
    for y, yd in zip(ys, desire_ys):
        np.testing.assert_array_equal(y, yd)
