#!/usr/bin/env python
import h5py
import kaldiio
import numpy as np
import pytest

from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.io_utils import SoundHDF5File
from espnet.utils.training.batchfy import make_batchset
from test.utils_test import make_dummy_json


@pytest.mark.parametrize('swap_io', [True, False])
def test_make_batchset(swap_io):
    dummy_json = make_dummy_json(128, [128, 512], [16, 128])
    # check w/o adaptive batch size
    batchset = make_batchset(dummy_json, 24, 2 ** 10, 2 ** 10,
                             min_batch_size=1, swap_io=swap_io)
    assert sum([len(batch) >= 1 for batch in batchset]) == len(batchset)
    print([len(batch) for batch in batchset])
    batchset = make_batchset(dummy_json, 24, 2 ** 10, 2 ** 10,
                             min_batch_size=10, swap_io=swap_io)
    assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)
    print([len(batch) for batch in batchset])

    # check w/ adaptive batch size
    batchset = make_batchset(dummy_json, 24, 256, 64,
                             min_batch_size=10, swap_io=swap_io)
    assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)
    print([len(batch) for batch in batchset])
    batchset = make_batchset(dummy_json, 24, 256, 64,
                             min_batch_size=10, swap_io=swap_io)
    assert sum([len(batch) >= 10 for batch in batchset]) == len(batchset)


@pytest.mark.parametrize('swap_io', [True, False])
def test_sortagrad(swap_io):
    dummy_json = make_dummy_json(128, [1, 700], [1, 700])
    if swap_io:
        batchset = make_batchset(dummy_json, 16, 2 ** 10, 2 ** 10, batch_sort_key="input",
                                 shortest_first=True, swap_io=True)
        key = 'output'
    else:
        batchset = make_batchset(dummy_json, 16, 2 ** 10, 2 ** 10, shortest_first=True)
        key = 'input'
    prev_start_ilen = batchset[0][0][1][key][0]['shape'][0]
    for batch in batchset:
        cur_start_ilen = batch[0][1][key][0]['shape'][0]
        assert cur_start_ilen >= prev_start_ilen
        prev_ilen = cur_start_ilen
        for sample in batch:
            cur_ilen = sample[1][key][0]['shape'][0]
            assert cur_ilen <= prev_ilen
            prev_ilen = cur_ilen
        prev_start_ilen = cur_start_ilen


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


@pytest.mark.parametrize('fmt', ['flac', 'wav'])
def test_sound_hdf5_file(tmpdir, fmt):
    valid = {'a': np.random.randint(-100, 100, 25, dtype=np.int16),
             'b': np.random.randint(-1000, 1000, 100, dtype=np.int16)}

    # Note: Specify the file format by extension
    p = tmpdir.join('test.{}.h5'.format(fmt)).strpath
    f = SoundHDF5File(p, 'a')

    for k, v in valid.items():
        f[k] = (v, 8000)

    for k, v in valid.items():
        t, r = f[k]
        assert r == 8000
        np.testing.assert_array_equal(t, v)


@pytest.mark.parametrize('typ', ['ctc', 'wer', 'cer', 'all'])
def test_error_calculator(tmpdir, typ):
    from espnet.nets.e2e_asr_common import ErrorCalculator
    space = "<space>"
    blank = "<blank>"
    char_list = [blank, space, 'a', 'e', 'i', 'o', 'u']
    ys_pad = [np.random.randint(0, len(char_list), x) for x in range(120, 150, 5)]
    ys_hat = [np.random.randint(0, len(char_list), x) for x in range(120, 150, 5)]
    if typ == 'ctc':
        cer, wer = False, False
    elif typ == 'wer':
        cer, wer = False, True
    elif typ == 'cer':
        cer, wer = True, False
    else:
        cer, wer = True, True

    ec = ErrorCalculator(char_list, space, blank,
                         cer, wer)

    if typ == 'ctc':
        cer_ctc_val = ec(ys_pad, ys_hat, is_ctc=True)
        _cer, _wer = ec(ys_pad, ys_hat)
        assert cer_ctc_val is not None
        assert _cer is None
        assert _wer is None
    elif typ == 'wer':
        _cer, _wer = ec(ys_pad, ys_hat)
        assert _cer is None
        assert _wer is not None
    elif typ == 'cer':
        _cer, _wer = ec(ys_pad, ys_hat)
        assert _cer is not None
        assert _wer is None
    else:
        cer_ctc_val = ec(ys_pad, ys_hat, is_ctc=True)
        _cer, _wer = ec(ys_pad, ys_hat)
        assert cer_ctc_val is not None
        assert _cer is not None
        assert _wer is not None


def test_error_calculator_nospace(tmpdir):
    from espnet.nets.e2e_asr_common import ErrorCalculator
    space = "<space>"
    blank = "<blank>"
    char_list = [blank, 'a', 'e', 'i', 'o', 'u']
    ys_pad = [np.random.randint(0, len(char_list), x) for x in range(120, 150, 5)]
    ys_hat = [np.random.randint(0, len(char_list), x) for x in range(120, 150, 5)]
    cer, wer = True, True

    ec = ErrorCalculator(char_list, space, blank,
                         cer, wer)

    cer_ctc_val = ec(ys_pad, ys_hat, is_ctc=True)
    _cer, _wer = ec(ys_pad, ys_hat)
    assert cer_ctc_val is not None
    assert _cer is not None
    assert _wer is not None
