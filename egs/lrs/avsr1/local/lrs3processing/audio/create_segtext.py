import sys, os

def main(textpath, segmentpath, savedir):
    filepath = textpath
    with open(filepath) as filelists:
        filelist = filelists.readlines()
    with open(segmentpath) as filelists:
        segmentinfo = filelists.readlines()
    filedict = {}
    segdict = {}

    for i in range(len(filelist)):
        filelist[i] = filelist[i].strip('\n')
    for i in range(len(segmentinfo)):
        segmentinfo[i] = segmentinfo[i].strip('\n')
    for i in range(len(filelist)):
        filedict.update({filelist[i].split(' ', 1)[0]: filelist[i].split(' ', 1)[1]})
    for i in range(len(segmentinfo)):
        segdict.update({segmentinfo[i].split(' ', 1)[0]: segmentinfo[i].split(' ', 1)[1]})
    filekeys = list(filedict.keys())
    for i in range(len(filekeys)):
        splitname = filekeys[i].split('/')
        textdir = os.path.join(savedir, splitname[0], splitname[1] + "p.txt")
        segtext = filedict[filekeys[i]]
        segtime = segdict[filekeys[i]]
        with open(textdir, "w") as textprocess:
                textprocess.write("Text:  " + segtext + '\n' + "time:  " + segtime)
                textprocess.close()

# hand over parameter overview
# sys.argv[1] = textpath (str): Path to file with text linked to pretrained files, 
#                               e.g. "/home/fabian/LRS3-Prep/LRS3/kaldi/pretrain/pretrain_text"
# sys.argv[2] = segmentpath (str): Directoy where the segmented audio files are stored
#                              e.g. "/home/fabian/LRS3-Prep/LRS3/LRS3pretraindata/audio/pretrainsegment/"
# sys.argv[3] = savedir (str): Directoy where the segmented audio files are stored
#                              e.g. "/home/fabian/LRS3-Prep/LRS3/LRS3pretraindata/audio/pretrainsegment/"

main(sys.argv[1],sys.argv[2], sys.argv[3])

