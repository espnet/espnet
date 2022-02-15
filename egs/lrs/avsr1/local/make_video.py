import json
import pickle, os
import sys
import torch
import numpy as np
from kaldiio import WriteHelper



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def make_ark(srcdir, savedir, nj):
    nj = int(nj)
    datatypes = os.listdir(srcdir)
    for datatype in datatypes:
        Resultsdir = os.path.join(savedir, datatype)
        if not os.path.exists(Resultsdir):
            os.makedirs(Resultsdir)
        print('make ark files')
        filedir = os.path.join(srcdir, datatype)
        filelist = os.listdir(filedir)
        filelists = list(split(filelist, nj))

        for i in range(len(filelists)):
            arksavedir = 'ark,scp:' + os.path.join(Resultsdir,
                                                   'feats_' + str(i) + '.ark') + ',' + os.path.join(
                Resultsdir, 'feats_' + str(i) + '.scp')
            with WriteHelper(arksavedir, compression_method=2) as writer:
                for filename in filelists[i]:
                    dsetsrcdata = torch.load(os.path.join(filedir, filename))
                    name = filename.split('.')[0]
                    print(os.path.join(filedir, filename))
                    if dsetsrcdata.detach().is_cuda:
                        writer(name, dsetsrcdata.detach().cpu().numpy())
                    else:
                        writer(name, dsetsrcdata.detach().numpy())




make_ark(sys.argv[1],sys.argv[2],sys.argv[3])
