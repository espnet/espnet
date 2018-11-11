#!/usr/bin/env python

import importlib

import numpy as np


def make_dummy_json(n_utts=10, ilen_range=[100, 300], olen_range=[10, 300]):
    idim = 83
    odim = 52
    ilens = np.random.randint(*ilen_range, n_utts)
    olens = np.random.randint(*olen_range, n_utts)
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
    for module in ["espnet.asr.asr_utils", "espnet.asr.asr_utils"]:
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
        print([len(batch) for batch in batchset])
