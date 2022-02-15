import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def cal_weights(attlogp, ctclogp, beam):
    ctclogp, bestid = torch.sort(ctclogp, descending=True)
    suminteratt = []
    suminterctc = []

    ####entropy
    attent = 0
    ctcent = 0
    for k in range(beam):
        attent = attent + torch.exp(attlogp[:, k]) * attlogp[:, k]
        ctcent = ctcent + torch.exp(ctclogp[k]) * ctclogp[k]
    attent = 1/(-attent)
    ctcent = 1/(-ctcent)
    sument = attent + ctcent
    entattw = attent / sument
    entctcw = ctcent / sument


    ####dispersion
    K = 15
    for l in range(K):
        sumtemp = 0
        for m in range(l + 1, K - 1):
            sumtemp = sumtemp + (attlogp[:, l] - attlogp[:, m])
        suminteratt.append(sumtemp)
        sumtemp = 0
        for m in range(l + 1, K - 1):
            sumtemp = sumtemp + (ctclogp[l] - ctclogp[m])
        suminterctc.append(sumtemp)
    scale = 2 / (K * (K - 1))
    attdis = scale * sum(suminteratt)
    ctcdis = scale * sum(suminterctc)
    sumdis = attdis + ctcdis
    disattw = attdis / sumdis
    disctcw = ctcdis / sumdis


    ####difference
    maxatt = max(attlogp[0, :])
    maxctc = max(ctclogp)
    attdiff = 0
    ctcdiff = 0
    for j in range(1, K):
        attdiff = attdiff + abs(maxatt - attlogp[:, j])
        ctcdiff = ctcdiff + abs(maxctc - ctclogp[j])
    attdiff = 1 / (K - 1) * attdiff
    ctcdiff = 1 / (K - 1) * ctcdiff
    sumdiff = attdiff + ctcdiff
    diffattw = attdiff / sumdiff
    diffctcw = ctcdiff / sumdiff

    attw = (entattw + diffattw + disattw) / 3.0
    ctcw = (entctcw + diffctcw + disctcw) / 3.0


    return attw, ctcw
