import os, sys, subprocess
import multiprocessing as mp

""" This code extract audio files from mp4 files."""
def main(savedir, segdir, ifmulticore=False, debug=False):
    """Args:
            savedir (str): The dir, where to save the extracted audio signal.
            segdir (str): The dir, where save the segment information txt file
            ifmulticore: If use multi processes.
            ifdebug: If use debug mode
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
    mp4list = []
    for i in range(len(filelist)):
        filelist[i] = filelist[i].split(' ')
        mp4list.append(filelist[i][1])

    mp4list = list(set(mp4list))
    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(savedir, j, debug) for j in mp4list]
        pool.map(product_helper, job_args)
    else:
        for j in mp4list:
            process(savedir, j, debug)


def product_helper(args):
    return process(*args)


def process(savedir, name, debug=False):
    """Extract the audio signal from each mp4 file.

       Args:
           savedir (str): The dir where save the mp3 files
           name (str): The name of the mp4 file.
           debug: If debug mode
    """
    namesplit = name.split('/')
    filename = namesplit[-2] + "_" + namesplit[-1].strip(".mp4") + ".wav"
    savedir = os.path.join(savedir, filename)
    try:
        cmd = "ffmpeg -i {} -vn -ac 1 -ar 16000 -ab 320k -f wav {}".format(name, savedir)
        ffmpeg_process = subprocess.run(cmd, shell=True,  stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
        backpass = ffmpeg_process.returncode
        if backpass == 0:
            stdout, stderr = (ffmpeg_process.stdout.decode("utf-8"), ffmpeg_process.stderr.decode("utf-8"))
            if debug == True:
                print("ffmpeg process for file {} finished successfully. Audio file saved under {}".format(name, savedir))
    
    except Exception as e:
        print(e)


# hand over parameter overview
# sys.argv[1] = savedir (str), Directory in which to save the audio signal
# sys.argv[2] = segdir (str), The dir of the file with the segmentation information path
# sys.argv[3] = ifmulticore (boolean), default False
# optional
# sys.argv[4] = debug (boolean), if debug should be used, default False


if len(sys.argv) > 4:
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
else:
    main(sys.argv[1],sys.argv[2],sys.argv[3])


