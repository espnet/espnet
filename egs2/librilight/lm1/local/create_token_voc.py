import os

import numpy as np

# This file creates a token vocabulary to be used with discrete speech tokens

# n_voc: number of discrete tokens, same as k used in k-means clustering for quantization
n_voc = 50
token_file = "data/tokens.txt"

with open(token_file, "w") as f:
    f.write("<blank>\n")
    f.write("<unk>\n")
    for i in range(n_voc):
        f.write("{}\n".format(i))
    f.write("<sos/eos>\n")
