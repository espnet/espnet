import math
import numpy as np
import torch
from torch import nn
import itertools


def tile(a, dim, n_tile):
    lists = []
    for i in range(a.size()[1]):
        frame = a[:, i, :].unsqueeze(1)
        temp = frame.repeat(1, n_tile[i], 1)
        lists.append(temp)
    output = torch.cat(lists, dim=1)

    return output


def dda(data, c2):
    c1 = data.size()[1]
    if c1 > c2:
        dx = c1
        dy = c2
    else:
        dx = c2
        dy = c1
    arr = np.zeros((dx))
    d = 2 * dy - dx
    dO = 2 * dy
    dNO = 2 * (dy - dx)
    y = 0
    x = 0
    err = d
    arr[x] = y
    for x in range(1, dx):
        if err <= 0:
            err = err + dO
        else:
            y = y + 1
            err = err + dNO
        arr[x] = y
    arr[arr > dy - 1] = dy - 1
    arr = arr.astype(int)
    Y = [(len(list(y))) for x, y in itertools.groupby(arr)]
    a = tile(data, 1, Y)

    return a
