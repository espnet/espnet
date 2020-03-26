import numpy as np


def make_dummy_json(n_utts=10, ilen_range=(100, 300), olen_range=(10, 300), idim=83, odim=52, num_inputs=1):
    ilens = np.random.randint(ilen_range[0], ilen_range[1], n_utts)
    olens = np.random.randint(olen_range[0], olen_range[1], n_utts)
    dummy_json = {}
    for idx in range(n_utts):
        input = []
        for input_idx in range(num_inputs):
            input += [{
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


def make_dummy_json_st(n_utts=10, ilen_range=(100, 300), olen_range=(10, 300),
                       olen_asr_range=(10, 300), idim=83, odim=52):
    ilens = np.random.randint(ilen_range[0], ilen_range[1], n_utts)
    olens = np.random.randint(olen_range[0], olen_range[1], n_utts)
    olens_asr = np.random.randint(olen_asr_range[0], olen_asr_range[1], n_utts)
    dummy_json = {}
    for idx in range(n_utts):
        input = [{
            "shape": [ilens[idx], idim]
        }]
        output = [
            {
                "shape": [olens[idx], odim]
            },
            {
                "shape": [olens_asr[idx], odim]
            }
        ]
        dummy_json["utt_%d" % idx] = {
            "input": input,
            "output": output
        }
    return dummy_json


def make_dummy_json_mt(n_utts=10, ilen_range=(100, 300), olen_range=(10, 300), idim=83, odim=52):
    ilens = np.random.randint(ilen_range[0], ilen_range[1], n_utts)
    olens = np.random.randint(olen_range[0], olen_range[1], n_utts)
    dummy_json = {}
    for idx in range(n_utts):
        output = [
            {
                "shape": [olens[idx], odim]
            },
            {
                "shape": [ilens[idx], idim]
            }]
        dummy_json["utt_%d" % idx] = {
            "output": output
        }
    return dummy_json
