#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import torch

from e2e_asr_attctc_th import pad_list
from e2e_asr_backtrans import Tacotron2
from e2e_asr_backtrans import Tacotron2Loss


def test_tacotron2():
    bs = 4
    idim = 10
    odim = 40
    ilens = np.sort(np.random.randint(1, idim, bs))[::-1]
    xs = pad_list([np.random.randint(0, idim, l) for l in ilens], 0)
    xs = torch.from_numpy(xs).long()
    olens = np.sort(np.random.randint(idim * 10, idim * 20, bs))[::-1]
    ys = pad_list([np.random.randn(l, odim) for l in olens], 0)
    ys = torch.from_numpy(ys).float()
    labels = ys.new_zeros((ys.size(0), ys.size(1)))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1

    model = Tacotron2(idim, odim)
    criterion = Tacotron2Loss(model)
    optimizer = torch.optim.Adam(model.parameters())

    after, before, logits = model(xs, ilens, ys)
    loss = criterion(xs, ilens, ys, labels, olens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        yhat, probs, att_ws = model.inference(xs[0][:ilens[0]])
        att_ws = model.calculate_all_attentions(xs, ilens, ys)
