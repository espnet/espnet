import os, sys
import multiprocessing as mp

def main(srcdir, videodir, dset, ifmulticore):
    if ifmulticore == 'true':
        ifmulticore = True
    else:
        ifmulticore = False
    savedir = srcdir
    exist = os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)

    segmentinfodir = os.path.join(srcdir, 'segments')
    with open(segmentinfodir) as filelists:
        segmentinfo = filelists.readlines()


    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, dset, savedir, videodir) for i in segmentinfo]
        pool.map(product_helper, job_args)
    else:
        for i in segmentinfo:
            set(i, dset, savedir, videodir)


def product_helper(args):
    return set(*args)


def set(file, s, savedir, videodir):
    splitdata = file.split(' ')
    name = splitdata[0]
    splitname = name.split('_')
    titel = name
    if 'LRS3' in videodir:
        mp4dir = '/'.join([videodir, s, splitname[0], splitname[1] + '.mp4'])
    else:
        mp4dir = '/'.join([videodir, s, splitname[1], splitname[2][:-1] + '.mp4'])
    segtime = ' '.join([splitdata[-2], splitdata[-1]])
    out = ' '.join([titel, mp4dir, segtime])
    outdir = savedir + '/seginfo.txt'

    with open(outdir, "a") as wav:
        wav.writelines(out)
        wav.close()


def remove(sub, s):
    return s.replace(sub, "", -1)




# hand over parameter overview
# sys.argv[1] = srcdir (str): Path to source dir
# sys.argv[2] = videodir (str): Path to Corpus
# sys.argv[3] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[4] = ifmulticore (str): If use multi processes.
main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

