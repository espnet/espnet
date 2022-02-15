import os, sys, math
import pydub
from pydub import AudioSegment
import itertools
import multiprocessing as mp

def seg(filelist, sourcedir, savedir):
    filelist = filelist.strip('\n')
    filelist = filelist.split(' ')
    ifsegment = len(filelist)

    filedir = filelist[0]
    seginfo = filelist[1:]
    audiodir = sourcedir + '/pretrain/' + filedir + '.wav'
    splitname = filedir.split('/')
    savedir1 = savedir + '/' + splitname[0]
    exist = os.path.exists(savedir1)
    try:
        os.mkdir(savedir1)
    except FileExistsError:
        pass

    Audio = AudioSegment.from_wav(audiodir)
    fs = Audio.frame_rate

    cutpoint = [int(float(x) * 1000) for x in seginfo]

    if cutpoint == []:
        finalsave = savedir1 + '/' + splitname[1] + '_00p' + '.wav'
        exist = os.path.isfile(finalsave)
        if exist is True:
            pass
        else:
            Audio.export(finalsave, format="wav")
            print(finalsave)


    for j in range(len(cutpoint) - 1):
        newAudio = Audio[cutpoint[j]: cutpoint[j + 1]]
        if ifsegment > 3:
            finalsave = savedir1 + '/' + splitname[1] + '_' + str(j + 1).zfill(2) + 'p' + '.wav'
            exist = os.path.isfile(finalsave)
            if exist is True:
                pass
            else:
                newAudio.export(finalsave, format="wav")
                print(finalsave)
        else:
            finalsave = savedir1 + '/' + splitname[1] + '_' + str(j).zfill(2) + 'p' + '.wav'
            exist = os.path.isfile(finalsave)
            if exist is True:
                pass
            else:
                newAudio.export(finalsave, format="wav")
                print(finalsave)


def product_helper(args):
    return seg(*args)

def main(sourcedir, filelist, dset, ifmulticore):
    ifmulticore = bool(ifmulticore)
    savedir = sourcedir
    filelistdir = filelist + '/' + dset + 'list'
    savedir = savedir + '/pretrainsegment'
    exist = os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)

    with open(filelistdir) as filelists:
        filelist = filelists.readlines()

    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, sourcedir, savedir) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            seg(i, sourcedir, savedir)

# hand over parameter overview
# sys.argv[1] = sourcedir (str)
# sys.argv[2] = filelistdir (str)
# sys.argv[3] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[4] = ifmulticore (str): If use multi processes.
main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

