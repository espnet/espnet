import os, sys
import itertools
#from segaudio import seg
import multiprocessing as mp

def main(savedir, sourcedir, filelistdir, dset, ifmulticore):
    ifmulticore = bool(ifmulticore)
    exist=os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)

    filelistdir = filelistdir # + '/' + dset + 'list'
    with open(filelistdir) as filelists:
        filelist = filelists.readlines()
    for i in range(len(filelist)):
        filelist[i]=filelist[i].strip('\n')

    audiodir=savedir+'/audio'
    existaudio=os.path.exists(audiodir)
    if existaudio ==1:
        pass
    else:
        os.mkdir(audiodir)
    filelist.sort()

    if dset == 'pretrain':
        subfiledir = 'pretrain'
    else:
        subfiledir = 'test'

    for i in filelist:
        namesplit = i.split('/')
        savedirs = audiodir + '/' + dset + '/' + namesplit[0]
        existdir = os.path.exists(savedirs)
        if existdir == 1:
            pass
        else:
            os.makedirs(savedirs)

    if ifmulticore is True:
        pool = mp.Pool(4)
        job_args = [(dset, audiodir, j, savedir, sourcedir, subfiledir) for j in filelist]
        pool.map(product_helper, job_args)
    else:
        for j in filelist:
            process(dset, audiodir, j, savedir, sourcedir, subfiledir)



def product_helper(args):
    return process(*args)

def process(s, audiodir, Name, savedir, sourcedir, subfiledir):
    name = Name.split(' ')[0]
    seginfo = Name.split(' ')[1:]
    inputFile = sourcedir + '/' + subfiledir + '/' + name + '.mp4'
    savecheckdir = audiodir + '/' + s + '/' + name + '.wav'
    check = os.path.exists(savecheckdir)
    if check is True:
        pass
    else:
        namesplit = name.split('/')
        savedir = audiodir + '/' + s + '/' + namesplit[0]
        outputname = namesplit[1] + ".wav"
        existdir = os.path.exists(audiodir + '/' + s)
        if existdir == 1:
            pass
        else:
            os.mkdir(audiodir + '/' + s)

        outFile = savedir + '/' + outputname
        try:
            cmd = "ffmpeg -y -i {} -vn -ac 1 -ar 16000 -ab 320k -f wav {}".format(inputFile, outFile)
            os.popen(cmd)
            print(outFile)
        except:
            print('error')
            with open(savedir + 'errorFilelist_' + s + '.txt', 'a+') as af:
                af.writelines(name + '\n')
            af.close()



# hand over parameter overview
# sys.argv[1] = savedir (str), Directory in which to save the audio signal
# sys.argv[2] = sourcedir (str), The dir of the files
# sys.argv[3] = filelistdir (str), Filelist with the files
# sys.argv[4] = dset (str), dataset
# sys.argv[5] = ifmulticore (boolean), if multicore processing
main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5])



