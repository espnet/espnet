import os, sys, math
from shutil import copyfile
import cv2
import numpy as np
import torch
import itertools
import multiprocessing as mp
def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video.append(frame)
        else:
            break
    cap.release()

    video = np.array(video)
    return video #[...,::-1]
def segvideo(sourcedir, filelist, savedir,dset, filedict):
    #print(filelist)
    mp4filedir = sourcedir + '/' + filelist + '.mp4'
    pics = extract_opencv(mp4filedir)[:, 55: 125, 45: 115]

    ####### if you want check the video segment, you should block this part #######
    pics = pics / 255.
    pics = [cv2.resize(pics[i,:,:], (96, 96)) for i in range(len(pics))]
    pics = np.array(pics)
    mean = 0.4161
    std = 0.1688
    pics = (pics - mean) / std
    ####### if you want check the video segment, you should block this part #######


    segdata = filedict[filelist]
    filelist = filelist.split('/')




    if segdata == '':
        ifsegment = 0
    else:
        segdata = segdata.split(' ')
        seginfo = [float(x) for x in segdata]
        ifsegment = len(seginfo)

    savedir1 = savedir
    exist = os.path.exists(savedir1)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir1)
    savedir2 = savedir1 + '/' + filelist[0]
    exist = os.path.exists(savedir2)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir2)




    if ifsegment > 2:
        cutpoint = [float(x) * 25 for x in seginfo]
        for j in range(len(cutpoint) - 1):
            savedir3 = savedir2 + '/' + filelist[1] + '_' + str(j + 1).zfill(2)
            exist = os.path.exists(savedir3 + '.pt')
            if exist is True:
                pass
            else:
                start = int(math.floor(cutpoint[j]))
                end = int(math.ceil(cutpoint[j + 1]))
                newVideo = pics[start: end]
                torch.save(newVideo, savedir3 + '.pt')
                print(savedir3)

    else:
        savedir3 = savedir2 + '/' + filelist[1] + '_' + str(0).zfill(2)
        exist = os.path.exists(savedir3 + '.pt')
        if exist is True:
            pass
        else:
            torch.save(pics, savedir3 + '.pt')
            print(savedir3)





def product_helper(args):
    return segvideo(*args)
def main(sourcedir, savedir,  dset, ifmulticore):
    ifmulticore = bool(ifmulticore)
    sourcedir = sourcedir + '/pretrain'
    filelistdir = savedir + '/audio/pretrain_segmentinfo/' + dset + 'list'

    savedir = savedir + '/video/' + dset + '_segment'
    exist = os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)



    with open(filelistdir) as filelists:
        filelist = filelists.readlines()

    filedict = {}
    for i in range(len(filelist)):
        filelist[i] = filelist[i].strip('\n')
        filelist[i] = filelist[i].split(' ')
        segdata = ' '.join(filelist[i][1:])
        filedict[filelist[i][0]] = segdata

    usefilelist = list(filedict.keys())

    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(sourcedir, i, savedir, dset, filedict) for i in usefilelist]
        pool.map(product_helper, job_args)
    else:
        for i in usefilelist:
            segvideo(sourcedir, i, savedir, dset, filedict)




main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
