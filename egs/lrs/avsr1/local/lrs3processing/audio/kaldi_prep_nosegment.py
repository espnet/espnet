import multiprocessing as mp
import os
import sys


def main(filelist, savedir, audiodir, sourcedir, dset, ifmulticore):
    ifmulticore = bool(ifmulticore)
    exist = os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)

    with open(filelist) as filelists:
        filelist = filelists.readlines()
    for i in range(len(filelist)):
        filelist[i] = filelist[i].strip("\n")
    filelist.sort()
    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, dset, savedir, sourcedir, audiodir) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            set(i, dset, savedir, sourcedir, audiodir)


def product_helper(args):
    return set(*args)


def set(file, s, savedir, sourcedir, audiodir):
    textdir = savedir + "/text"
    utt2spkdir = savedir + "/utt2spk"
    wavdirs = savedir + "/wav.scp"
    notexist = savedir + "/notexist"
    # wavfiledir=audiodir +  '/'

    name = file.split("/")[0] + "_" + file.split("/")[1]
    textsrcdir = sourcedir + "/" + s + "/" + file + ".txt"
    with open(textsrcdir) as filelists:
        text = filelists.readlines()
    text = text[0].strip("\n").replace("Text:  ", "")

    wavdir = audiodir + "/" + file + ".wav"

    with open(textdir, "a") as textprocess:
        textprocess.writelines(name + " " + text + "\n")
        textprocess.close()
    with open(utt2spkdir, "a") as utt:
        utt.writelines(name + " " + name + "\n")
        utt.close()

    with open(wavdirs, "a") as wav:
        wav.writelines(name + " " + wavdir + "\n")
        wav.close()


def remove(sub, s):
    return s.replace(sub, "", -1)


# hand over parameter overview
# sys.argv[1] = filelist (str): Path to filelist
# sys.argv[2] = savedir (str): Directoy where to store the kaldi files
# sys.argv[3] = audiodir (str): Directory of the audio files
#               (seperated from video files)
# sys.argv[4] = sourcedir (str): LRS3 save directory
# sys.argv[5] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[6] = ifmulticore (str): If use multi processes.
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
