import multiprocessing as mp
import os
from pydub import AudioSegment
import sys


def seg(filelist, sourcedir, dset, savedir):
    filelist = filelist.strip("\n")
    filelist = filelist.split(" ")
    starttime = filelist[-2]
    endtime = filelist[-1]
    filename = filelist[0]

    audiodir = os.path.join(sourcedir, dset, filename + ".wav")
    finalsave = os.path.join(savedir, filename + ".wav")

    Audio = AudioSegment.from_wav(audiodir)
    startpoint = int(float(starttime) * 1000)
    endpoint = int(float(endtime) * 1000)

    cutAudio = Audio[startpoint:endpoint]

    cutAudio.export(finalsave, format="wav")
    print(finalsave)


def product_helper(args):
    return seg(*args)


def main(sourcedir, filedir, dset, ifmulticore):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False
    savedir = sourcedir
    filedir = os.path.join(filedir, dset + "_aug", "segments")
    savedir = os.path.join(savedir, dset + "_aug")
    exist = os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.makedirs(savedir)

    with open(filedir) as filelists:
        filelist = filelists.readlines()

    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, sourcedir, dset, savedir) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            seg(i, sourcedir, dset, savedir)


# hand over parameter overview
# sys.argv[1] = sourcedir (str)
# sys.argv[2] = filedir (str)
# sys.argv[3] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[4] = ifmulticore (str): If use multi processes.
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
