import os, sys
from scipy.io import wavfile
import multiprocessing as mp

""" This code crop the audio file by using the segment information."""

def seg(savedir, seginfo, debug=False):
    """Crop the audio files into fixed intervals.

    Args:
            seginfo (str): The dir of the file with the segmentation information path
            savedir (str): The dir, where to save the extracted audio signal.
            debug: If debug mode
    """
    # get information from seginfo file, split name, finally extract segment start and end time for file
    seginfo = seginfo.split(" ")
    savesplitwavdir = seginfo[0]
    namesplit = seginfo[1].split('/')
    filename = namesplit[-2] + "_" + namesplit[-1].strip(".mp4") + ".wav"
    fullwavdir = os.path.join(savedir, filename)
    segstarttime = float(seginfo[2])
    segendtime = float(seginfo[3])
    segtimes = [segstarttime, segendtime]
    samplerate, data = wavfile.read(fullwavdir)

    # calculate the discrete sample number for the cutpoints
    cutpoint = [int(float(x) * samplerate) for x in segtimes]

    # cut audio signal and write to new file
    newAudio = data[cutpoint[0]: cutpoint[1]]
    try: 
        wavfile.write(savesplitwavdir, samplerate, newAudio)
        if debug == True:
            print("Audio file {} successfully splitted into file {}".format(fullwavdir, savesplitwavdir))
    except Exception as e:
        print(e)


def product_helper(args):
    return seg(*args)


def main(savedir, segdir, ifmulticore=False, debug=False):
    """Crop the audio files into fixed interval.

    Args:
            savedir (str): The dir, where to save the extracted audio signal.
            segdir (str): The dir of the file with the segmentation information path
            ifmulticore: If use multi processes.
            debug: If debug mode should be used		
    """
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False
    if debug == "true":
        debug = True
    else: 
        debug = False
    
    with open(os.path.join(segdir, "seginfo.txt")) as filelists:
        filelist = filelists.readlines()
    fullmp3list = []
    for i in range(len(filelist)):
        splitfile = filelist[i].split(' ')
        namesplit = splitfile[1].split('/')
        filename = namesplit[-2] + "_" + namesplit[-1].strip(".mp4") + ".wav"
        fullmp3list.append(os.path.join(savedir, filename))

    fullmp3list = list(set(fullmp3list))

    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(savedir, i, debug) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            seg(savedir, i, debug)
    for i in fullmp3list:
        os.remove(i)

# hand over parameter overview
# sys.argv[1] = savedir (str), Directory in which to save the extracted audio signal.
# sys.argv[2] = segdir (str), The dir of the file with the segmentation information path
# sys.argv[3] = ifmulticore (boolean), default False
# optional
# sys.argv[4] = debug (boolean), if debug should be used, default False


if len(sys.argv) > 4:
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
else:
    main(sys.argv[1],sys.argv[2],sys.argv[3])
