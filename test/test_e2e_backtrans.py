#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import torch

from e2e_asr_attctc_th import pad_list
from e2e_asr_backtrans import Tacotron2


def test_tacotron2():
    bs = 4
    idim = 10
    edim = 128
    odim = 40
    ilens = np.sort(np.random.randint(1, idim, bs))[::-1]
    xs = pad_list([np.random.randint(0, idim, l) for l in ilens], 0)
    xs = torch.from_numpy(xs).long()
    olens = np.sort(np.random.randint(idim * 10, idim * 20, bs))[::-1]
    ys = pad_list([np.random.randn(l, odim) for l in olens], 0)
    ys = torch.from_numpy(ys).float()

    model = Tacotron2(idim, edim, odim, maxlenratio=10)
    optimizer = torch.optim.Adam(model.parameters())

    post_outs, outs, probs, olens = model(xs, ilens, ys, olens)
    mse, bce = model.loss(ys, post_outs, outs, probs, olens)
    loss = mse + bce
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    yhat, probs, att_ws = model.inference(xs[0][:ilens[0]])
    print(yhat)
    print(probs)
    print(att_ws)
