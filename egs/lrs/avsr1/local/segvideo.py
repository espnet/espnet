import cv2
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys
import torch


def extract_pretrain_opencv(mp4filedir, csvname, segmentslist, corpus):
    """Using cv2 extract video frames.

    Args:
        mp4filedir (str): The video file name.
        csvname (str): The csv file saved face recog info
        segmentslist (list): The list saved segment info
        corpus (str): With LRS2 or LRS3 corpus

    """
    video = []
    cap = cv2.VideoCapture(mp4filedir)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    CSV = pd.read_csv(csvname)
    head = CSV.axes[1].values
    for id in range(len(head)):
        if head[id] == "confidence":
            confid = id
        elif head[id] == "x_48":
            xstart = id
        elif head[id] == "x_67":
            xend = id + 1
        elif head[id] == "y_48":
            ystart = id
        elif head[id] == "y_67":
            yend = id + 1
        elif head[id] == "AU12_r":
            AU12id = id
        elif head[id] == "AU15_r":
            AU15id = id
        elif head[id] == "AU17_r":
            AU17id = id
        elif head[id] == "AU23_r":
            AU23id = id
        elif head[id] == "AU25_r":
            AU25id = id
        elif head[id] == "AU26_r":
            AU26id = id
    conf = CSV.values[:, confid]
    AUs = []
    for ids in [AU12id, AU15id, AU17id, AU23id, AU25id, AU26id]:
        AU = np.expand_dims(CSV.values[:, ids], axis=1)
        AUs.append(AU)

    AUdata = np.concatenate((AUs), axis=1)

    cropframe = []
    x = CSV.values[:, xstart:xend]
    y = CSV.values[:, ystart:yend]
    for i in range(len(video)):
        frame = video[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        a = max(x[i, :])
        b = max(y[i, :])
        c = min(x[i, :])
        d = min(y[i, :])

        if a == c or c == d:
            d = 55
            b = 125
            c = 45
            a = 115

        midx = int((d + b) / 2.0)
        midy = int((a + c) / 2.0)
        W = 70
        H = 70
        newd = midx - W / 2
        newb = midx + W / 2
        newc = midy - H / 2
        newa = midy + H / 2
        if newd < 0:
            newd = 0
            newb = 0 + W
        elif newb > 160:
            newb = 160
            newd = 160 - W
        elif newc < 0:
            newc = 0
            newa = 0 + H
        elif newa > 160:
            newa = 160
            newc = 160 - H

        outimage = gray[int(newd) : int(newb), int(newc) : int(newa)]
        if list(outimage.shape) == [70, 70]:
            cropframe.append(gray[int(newd) : int(newb), int(newc) : int(newa)])
        else:
            cropframe.append(gray[55:125, 45:115])

    cropframe = np.array(cropframe)
    cropframe = cropframe / 255.0
    cropframe = [
        cv2.resize(cropframe[i, :, :], (96, 96)) for i in range(len(cropframe))
    ]
    cropframe = np.array(cropframe)
    mean = 0.4161
    std = 0.1688
    cropframe = (cropframe - mean) / std
    output = {}
    for k in range(len(segmentslist)):
        if corpus == "LRS2":
            filename = segmentslist[k].split(" ")[0].split("/")[-1].strip(".wav")
        else:
            filename = segmentslist[k].split(" ")[0]
        output.update({filename: {}})
        seginfo = [segmentslist[k].split(" ")[1], segmentslist[k].split(" ")[2]]
        cutpoint = [float(x) * 25 for x in seginfo]
        start = int(np.floor(cutpoint[0]))
        end = int(np.ceil(cutpoint[1]))
        output[filename].update({"conf": conf[start:end]})
        output[filename].update({"AU": AUdata[start:end, :]})
        output[filename].update({"frames": cropframe[start:end, :, :]})

    return output


def segpretrainvideo(segdict, savedir, csvdir, corpus):
    """Segment video files, save data in pt files.

    Args:
        segdict (dict): The dictionary save the segment information
        savedir (str): Save the segmented video files.
        csvdir (str): The dir of csv File, which contain Face recognition information
        corpus (str): With LRS2 or LRS3 corpus

    """

    mp4filedir = list(segdict.keys())[0]
    segmentslist = segdict[mp4filedir]
    filename = "/".join(
        [mp4filedir.split("/")[-2], mp4filedir.split("/")[-1].split(".")[0]]
    )
    csvdirfile = os.path.join(csvdir, filename + ".csv")
    segmented = extract_pretrain_opencv(mp4filedir, csvdirfile, segmentslist, corpus)

    filekeys = segmented.keys()
    for file in filekeys:
        confdir = os.path.join(savedir, "Conf", file + ".pt")
        AUdir = os.path.join(savedir, "AUs", file + ".pt")
        Picdir = os.path.join(savedir, "Pics", file + ".pt")

        torch.save(
            segmented[file]["conf"], confdir, _use_new_zipfile_serialization=False
        )
        torch.save(segmented[file]["AU"], AUdir, _use_new_zipfile_serialization=False)
        torch.save(
            segmented[file]["frames"], Picdir, _use_new_zipfile_serialization=False
        )


def extract_opencv(filename, csvname):
    """Using cv2 extract video frames.

    Args:
        filename (str): The video file name.
        csvname (str): The csv file saved face recog info

    """
    video = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    CSV = pd.read_csv(csvname)
    head = CSV.axes[1].values
    for id in range(len(head)):
        if head[id] == "confidence":
            confid = id
        elif head[id] == "x_48":
            xstart = id
        elif head[id] == "x_67":
            xend = id + 1
        elif head[id] == "y_48":
            ystart = id
        elif head[id] == "y_67":
            yend = id + 1
        elif head[id] == "AU12_r":
            AU12id = id
        elif head[id] == "AU15_r":
            AU15id = id
        elif head[id] == "AU17_r":
            AU17id = id
        elif head[id] == "AU23_r":
            AU23id = id
        elif head[id] == "AU25_r":
            AU25id = id
        elif head[id] == "AU26_r":
            AU26id = id
    conf = CSV.values[:, confid]
    AUs = []
    for ids in [AU12id, AU15id, AU17id, AU23id, AU25id, AU26id]:
        AU = np.expand_dims(CSV.values[:, ids], axis=1)
        AUs.append(AU)

    AUdata = np.concatenate((AUs), axis=1)

    cropframe = []
    x = CSV.values[:, xstart:xend]
    y = CSV.values[:, ystart:yend]
    for i in range(len(video)):
        frame = video[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        a = max(x[i, :])
        b = max(y[i, :])
        c = min(x[i, :])
        d = min(y[i, :])

        if a == c or c == d:
            d = 55
            b = 125
            c = 45
            a = 115

        midx = int((d + b) / 2.0)
        midy = int((a + c) / 2.0)
        W = 70
        H = 70
        newd = midx - W / 2
        newb = midx + W / 2
        newc = midy - H / 2
        newa = midy + H / 2
        if newd < 0:
            newd = 0
            newb = 0 + W
        elif newb > 160:
            newb = 160
            newd = 160 - W
        elif newc < 0:
            newc = 0
            newa = 0 + H
        elif newa > 160:
            newa = 160
            newc = 160 - H

        outimage = gray[int(newd) : int(newb), int(newc) : int(newa)]
        if list(outimage.shape) == [70, 70]:
            cropframe.append(gray[int(newd) : int(newb), int(newc) : int(newa)])
        else:
            cropframe.append(gray[55:125, 45:115])

    cropframe = np.array(cropframe)
    cropframe = cropframe / 255.0
    cropframe = [
        cv2.resize(cropframe[i, :, :], (96, 96)) for i in range(len(cropframe))
    ]
    cropframe = np.array(cropframe)
    mean = 0.4161
    std = 0.1688
    cropframe = (cropframe - mean) / std

    return conf, AUdata, cropframe


def segvideo(sourcedir, filelist, savedir, csvdir, dset):
    """Segment video files, save data in pt files.

    Args:
        sourcedir (str): The LRS2 dataset dir.
        filelist (str): The dir of the mp4 file, it should be
                        like '5535415699068794046/00001'
        savedir (str): Save the segmented video files.
        csvdir (str): The dir of csv File, which contain Face recognition information

    """
    mp4filedir = sourcedir + "/" + filelist + ".mp4"
    print(mp4filedir)
    csvdir = os.path.join(csvdir, filelist + ".csv")
    filelist = filelist.split("/")
    conf, AUs, pics = extract_opencv(mp4filedir, csvdir)

    if dset == "pretrain":
        confdir = os.path.join(
            savedir, "Conf", "LRS2_" + filelist[0] + "_" + filelist[1] + "p.pt"
        )
        AUdir = os.path.join(
            savedir, "AUs", "LRS2_" + filelist[0] + "_" + filelist[1] + "p.pt"
        )
        Picdir = os.path.join(
            savedir, "Pics", "LRS2_" + filelist[0] + "_" + filelist[1] + "p.pt"
        )
    else:
        confdir = os.path.join(
            savedir, "Conf", "LRS2_" + filelist[0] + "_" + filelist[1] + "m.pt"
        )
        AUdir = os.path.join(
            savedir, "AUs", "LRS2_" + filelist[0] + "_" + filelist[1] + "m.pt"
        )
        Picdir = os.path.join(
            savedir, "Pics", "LRS2_" + filelist[0] + "_" + filelist[1] + "m.pt"
        )

    torch.save(conf, confdir)
    torch.save(AUs, AUdir)
    torch.save(pics, Picdir)


def product_helper(args):
    return segvideo(*args)


def product_helperpretrain(args):
    return segpretrainvideo(*args)


def main(sourcedir, savedir, csvdir, audiorefdir, dset, corpus, ifsegment, ifmulticore):
    """Segment video files, save data in pt files.

    Args:
        sourcedir (str): The LRS2 dataset dir.
        savedir (str): Save the segmented video data.
        csvdir (str): The dir of csv File, which contain Face recognition information
        audiorefdir (str): The dir which saves the audio Info
        dset (str): Which set. There are pretrain, Train, Val, Test set.
        ifsegment: If segmentation for pretrain set is used
        ifmulticore: If use multi processes.

    """
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False
    if ifsegment == "true":
        ifsegment = True
    else:
        ifsegment = False

    if ifsegment is False:
        filelistdir = os.path.join(audiorefdir, "text")
        with open(filelistdir) as filelists:
            filelist = filelists.readlines()

        for i in range(len(filelist)):
            filelist[i] = filelist[i].strip("\n")
            filelist[i] = filelist[i].split(" ")[0]
            filelist[i] = "/".join(
                [filelist[i].split("_")[1], filelist[i].split("_")[2][:-1]]
            )
        if ifmulticore is True:
            pool = mp.Pool()
            job_args = [(sourcedir, i, savedir, csvdir, dset) for i in filelist]
            pool.map(product_helper, job_args)
        else:
            for i in filelist:
                segvideo(sourcedir, i, savedir, csvdir, dset)
    else:
        filelistdir = os.path.join(audiorefdir, "text")
        with open(filelistdir) as filelists:
            filelist = filelists.readlines()
        nosegmentlist = []
        for i in range(len(filelist)):
            filelist[i] = filelist[i].strip("\n")
            filelist[i] = filelist[i].split(" ")[0]
            filelist[i] = filelist[i].split("_")
            if len(filelist[i]) == 5:
                pass
            else:
                nosegmentlist.append(("/".join([filelist[i][1], filelist[i][2][:-1]])))
        if dset == "pretrain":
            segfiledir = os.path.join(audiorefdir, "seginfo.txt")
            segdict = {}
            with open(segfiledir) as segfilelists:
                seglist = segfilelists.readlines()
            for i in seglist:
                segdict.update({i.split(" ")[1]: []})
            for j in seglist:
                splittext = j.split(" ")
                name = splittext[1]

                values = " ".join([splittext[0], splittext[2], splittext[3]])
                segdict[name].append(values)
            segmentdictkey = segdict.keys()
            if corpus == "LRS2":
                if ifmulticore is True:
                    pool = mp.Pool()
                    job_args = [
                        ({i: segdict[i]}, savedir, csvdir, corpus)
                        for i in segmentdictkey
                    ]
                    pool.map(product_helperpretrain, job_args)
                else:
                    for i in segmentdictkey:
                        segpretrainvideo({i: segdict[i]}, savedir, csvdir, corpus)
                if ifmulticore is True:
                    pool = mp.Pool()
                    job_args = [
                        (sourcedir, i, savedir, csvdir, dset) for i in nosegmentlist
                    ]
                    pool.map(product_helper, job_args)
                else:
                    for i in nosegmentlist:
                        segvideo(sourcedir, i, savedir, csvdir, dset)
            else:
                if ifmulticore is True:
                    pool = mp.Pool()
                    job_args = [
                        ({i: segdict[i]}, savedir, csvdir, corpus)
                        for i in segmentdictkey
                    ]
                    pool.map(product_helperpretrain, job_args)
                else:
                    for i in segmentdictkey:
                        segpretrainvideo({i: segdict[i]}, savedir, csvdir, corpus)

        else:
            if ifmulticore is True:
                pool = mp.Pool()
                job_args = [
                    (sourcedir, i, savedir, csvdir, dset) for i in nosegmentlist
                ]
                pool.map(product_helper, job_args)
            else:
                for i in nosegmentlist:
                    segvideo(sourcedir, i, savedir, csvdir, dset)


# hand over parameter overview
# sys.argv[1] = sourcedir (str): The dataset dir
# sys.argv[2] = savedir (str): Save the segmented video data.
# sys.argv[3] = csvdir (str): The dir of csv File, which contain
#                             Face recognition information
# sys.argv[4] = audiorefdir (str): The dir which saves the audio Info
# sys.argv[5] = dset (str): Which set. There are pretrain, Train, Val, Test set.
# sys.argv[6] = corpus (str): Corpus name, LRS2 or LRS3
# sys.argv[7] = ifsegment: If segmentation for pretrain set is used
# sys.argv[8] = ifmulticore: If use multi processes.

main(
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
    sys.argv[7],
    sys.argv[8],
)
