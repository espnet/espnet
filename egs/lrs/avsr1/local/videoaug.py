import cv2
import numpy as np
import os
import sys
from vidaug import augmentors as va


def videoaugmentation(filelist, srcdir, savedir, noisetype):
    savedir = os.path.join(savedir, noisetype)
    srcdir = os.path.join(srcdir, "data", "lrs2_v1", "mvlrs_v1", "main")
    lists = []
    with open(filelist) as fls:
        filelists = fls.readlines()
    for i in range(len(filelists)):
        lists.append(filelists[i].strip("\n") + ".mp4")

    for i in lists:
        videodir = os.path.join(srcdir, i)
        filename = os.path.join(savedir, i.split("/")[0])
        if not os.path.exists(filename):
            os.makedirs(filename)
        savevideodir = os.path.join(savedir, i)
        cap = cv2.VideoCapture(videodir)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        frames = np.asarray(frames)

        # sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        if noisetype == "blur":
            seq = va.Sequential([va.GaussianBlur(1.2), va.Add()])
        else:
            seq = va.Sequential(
                [
                    va.Salt(),
                    va.Pepper(),
                ]
            )
        frames = seq(frames)

        frame_width = frames[0].shape[0]
        frame_height = frames[0].shape[1]
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(savevideodir, fourcc, 25.0, (frame_width, frame_height))
        for i in frames:
            out.write(i)
        out.release()


# hand over parameter overview
# sys.argv[1] = filelist (str), Directory to save the file list, which are files augmentated
# sys.argv[2] = srcdir (str), Directory where save the dataset
# sys.argv[3] = savedir (str), Directory to save augmented files
# sys.argv[4] = noisetype (str), Video feature directory


videoaugmentation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
