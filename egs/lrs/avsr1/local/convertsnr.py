import numpy as np
import multiprocessing as mp
import os
from scipy.io import loadmat
import sys
import torch


def meansnr(filename, savedir, srcdir):
    try:
        SNR = loadmat(os.path.join(srcdir, filename))
        data = SNR["xi_hat"]
        datamean = np.mean(data, axis=1)
        datamean = torch.FloatTensor(datamean)
        filename = filename.split(".")[0]
        torch.save(
            datamean,
            os.path.join(savedir, filename + ".pt"),
            _use_new_zipfile_serialization=False,
        )
    except AssertionError as error:
        print(error)
        print('Pass this file')


def product_helper(args):
    return meansnr(*args)


def main(srcdir, savedir, ifmulticore):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False

    filelist = os.listdir(srcdir)

    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, savedir, srcdir) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            meansnr(i, savedir, srcdir)


# hand over parameter overview
# sys.argv[1] = sourcedir (str): The SNR datadir created by DeepXI
# sys.argv[2] = savedir (str): Save directory of the converted SNR (.pt files)
# sys.argv[7] = ifmulticore: If use multi processes.
main(sys.argv[1], sys.argv[2], sys.argv[3])
