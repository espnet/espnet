# coding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import importlib
import argparse

import pytest
import numpy


args = argparse.Namespace(
    elayers = 4,
    subsample = "1_2_2_1_1",
    etype = "vggblstmp",
    eunits = 100,
    eprojs = 100,
    dlayers=1,
    dunits=300,
    atype="location",
    aconv_chans=10,
    aconv_filts=100,
    mtlalpha=0.5,
    adim=320,
    dropout_rate=0.0,
    beam_size=3,
    penalty=0.5,
    maxlenratio=1.0,
    minlenratio=0.0,
    verbose = True,
    char_list = [u"あ", u"い", u"う", u"え", u"お"],
    outdir = None
)



def test_model_trainable_and_decodable():
    for m_str in ["e2e_asr_attctc", "e2e_asr_attctc_th"]:
        try:
            import torch
        except:
            if m_str[-3:] == "_th":
                pytest.skip("pytorch is not installed")

        m = importlib.import_module(m_str)
        model = m.Loss(m.E2E(40, 5, args), 0.5)
        out_data = "1 2 3 4"
        data = [
            ("aaa", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid=out_data)),
            ("bbb", dict(feat=numpy.random.randn(200, 40).astype(numpy.float32), tokenid=out_data))
        ]
        attn_loss = model(data)
        attn_loss.backward() # trainable

        in_data = data[0][1]["feat"]
        y = model.predictor.recognize(in_data, args, args.char_list) # decodable

