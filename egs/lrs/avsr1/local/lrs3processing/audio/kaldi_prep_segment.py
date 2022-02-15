import os, sys, wave, contextlib
import re, string
import itertools
import multiprocessing as mp

def main(videodir, filelistdir, savedir, dset, ifmulticore):
    if ifmulticore == 'true':
        ifmulticore = True
    else:
        ifmulticore = False
    exist=os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)
    
    segmentinfodir = os.path.join(filelistdir, 'pretrain_timeinfo')
    segmenttextdir = os.path.join(filelistdir, 'pretrain_text')
    with open(segmentinfodir) as filelists:
        segmentinfo = filelists.readlines()
    with open(segmenttextdir) as filelists:
        segmenttextinfo = filelists.readlines()
    segmentinfodict = {}
    for i in range(len(segmentinfo)):
        segmentinfo[i] = segmentinfo[i].strip('\n')
        segmentinfo[i] = segmentinfo[i].split(' ')
        time = ' '.join(segmentinfo[i][1 :])
        segmentinfodict.update({segmentinfo[i][0]: {}})
        segmentinfodict[segmentinfo[i][0]].update({'Time': time})
    for i in range(len(segmenttextinfo)):
        segmenttextinfo[i] = segmenttextinfo[i].strip('\n')
        segmenttextinfo[i] = segmenttextinfo[i].split(' ')
        Text = ' '.join(segmenttextinfo[i][1 :])
        segmentinfodict[segmenttextinfo[i][0]].update({'Text': Text})
    filelist = list(segmentinfodict.keys())
    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, dset, savedir, segmentinfodict, videodir) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            set(i, dset, savedir, segmentinfodict, videodir)

def product_helper(args):
    return set(*args)

def set(file, s, savedir, segmentinfodict, videodir):
    textdir= savedir + '/text'
    utt2spkdir=savedir + '/utt2spk'
    wavdirs=savedir + '/wav.scp'
    segdirs=savedir + '/segments'


    starttime = float(segmentinfodict[file]['Time'].split(' ')[0])
    endtime = float(segmentinfodict[file]['Time'].split(' ')[1])
    command1 = "ffmpeg -y -i"
    command2 = "-vn -ac 1 -ar 16000 -ab 320k -f wav /tmp/tmp.$$; cat /tmp/tmp.$$ |\n"
    splitname = file.split("/")
    mp4dir = '/'.join([videodir, s, splitname[0], splitname[1].split('_')[0] + '.mp4'])
    Title = '_'.join([splitname[0], splitname[1] + 'p'])
    spkerid = '_'.join([splitname[0], splitname[1].split('_')[0]])
    texttxt = " ".join([Title, segmentinfodict[file]['Text']])
    wavtxt = " ".join([Title, command1, mp4dir, command2])
    segtxt = " ".join([Title, Title, str(starttime), str(endtime)])
    utttxt = Title + ' ' + spkerid
    with open(textdir, "a") as textprocess:
        textprocess.writelines(texttxt + '\n')
        textprocess.close()
    with open(utt2spkdir, "a") as utt:
        utt.writelines(utttxt + '\n')
        utt.close()
    with open(wavdirs, "a") as wav:
        wav.writelines(wavtxt)
        wav.close()
    with open(segdirs, "a") as wav:
        wav.writelines(segtxt + '\n')
        wav.close()


def remove(sub,s):
    return s.replace(sub, "", -1)


# hand over parameter overview
# sys.argv[1] = videodir (str): Path to Corpus
# sys.argv[2] = filelistdir (str): Path to filelist
# sys.argv[3] = savedir (str): Directoy where to store the kaldi files
# sys.argv[4] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[5] = ifmulticore (str): If use multi processes.
main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

