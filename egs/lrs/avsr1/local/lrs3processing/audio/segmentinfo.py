import multiprocessing as mp
import os
import sys


def remove(sub, s):
    return s.replace(sub, "", -1)


def seg(seginfodir, filelist, savedir, dset):
    datadir = seginfodir + "/" + filelist + ".txt"
    print(datadir)
    with open(datadir) as filediscr:
        info = filediscr.readlines()
    info[0] = remove("Text:  ", info[0])
    starttime = info[4].split(" ")[1]
    endtime = float(info[-1].split(" ")[2])
    if endtime > 6:
        cutpoint = []
        timer = 5
        for j in range(4, len(info)):
            wordendtime = float(info[j].split(" ")[2])
            if wordendtime / timer > 1:
                cutpoint.append(j)
                timer = timer + 5
        cuttime = []
        for k in range(len(cutpoint)):
            cuttime.append(info[cutpoint[k]].split(" ")[2])

        if cuttime == []:
            ifseg = False
        else:
            ifseg = True
        cuttime = [starttime] + cuttime + [str(endtime)]
        cuttime = " ".join(cuttime)

        text = []
        for m in range(4, len(info)):
            text.append(info[m].split(" ")[0])
        cuttext = [x - 4 for x in cutpoint]
        cuttext.append(len(text))
        textlen = []
        for n in range(1, len(cuttext)):
            textlen.append(cuttext[n] - cuttext[n - 1])
        textlen = [cuttext[0] + 1] + textlen
        textlen[-1] = textlen[-1] - 1
        textseg = []
        for p in range(len(textlen)):
            temp = []
            for q in range(textlen[p]):
                temp.append(text[q])
            textseg.append(" ".join(temp))
            text[0 : textlen[p]] = []

        timeseg = []
        cuttimesplit = cuttime.split(" ")
        for j in range(len(cuttimesplit) - 1):
            timeseg.append(" ".join((cuttimesplit[j], cuttimesplit[j + 1])))

        if ifseg == True:
            for s in range(len(textseg)):
                if textseg[s] == "":
                    w = 1
                else:
                    with open(savedir + "/" + dset + "_text", "a") as text:
                        text.writelines(
                            filelist
                            + "_"
                            + str(s + 1).zfill(2)
                            + " "
                            + textseg[s]
                            + "\n"
                        )
                        text.close()
                    with open(savedir + "/" + dset + "_timeinfo", "a") as time:
                        time.writelines(
                            filelist
                            + "_"
                            + str(s + 1).zfill(2)
                            + " "
                            + timeseg[s]
                            + "\n"
                        )
                        time.close()
            if textseg[-1] == "":
                splitcut = cuttime.split(" ")
                del splitcut[-1]
                cuttime = " ".join(splitcut)

            with open(savedir + "/" + dset + "list", "a") as seg:
                seg.writelines(filelist + " " + cuttime + "\n")
                seg.close()
        else:
            cuttime = [starttime] + [str(endtime)]
            cuttime = " ".join(cuttime)
            with open(savedir + "/" + dset + "_text", "a") as text:
                text.writelines(filelist + "_" + str(0).zfill(2) + " " + info[0])
                text.close()
            with open(savedir + "/" + dset + "_timeinfo", "a") as time:
                time.writelines(filelist + "_" + str(0).zfill(2) + " " + cuttime + "\n")
                time.close()
            with open(savedir + "/" + dset + "list", "a") as seg:
                seg.writelines(filelist + " " + cuttime + "\n")
                seg.close()
    else:
        cuttime = [starttime] + [str(endtime)]
        cuttime = " ".join(cuttime)
        with open(savedir + "/" + dset + "_text", "a") as text:
            text.writelines(filelist + "_" + str(0).zfill(2) + " " + info[0])
            text.close()
        with open(savedir + "/" + dset + "_timeinfo", "a") as time:
            time.writelines(filelist + "_" + str(0).zfill(2) + " " + cuttime + "\n")
            time.close()
        with open(savedir + "/" + dset + "list", "a") as seg:
            seg.writelines(filelist + " " + cuttime + "\n")
            seg.close()


def product_helper(args):
    return seg(*args)


def main(sourcedir, savedir, filelistdir, dset, ifmulticore):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False
    ifmulticore = bool(ifmulticore)
    seginfodir = sourcedir + "/" + dset
    filelist = filelistdir + "/Filelist_" + dset
    savedir = savedir + "/audio/" + dset + "_segmentinfo"
    exist = os.path.exists(savedir)
    if exist == 1:
        pass
    else:
        os.mkdir(savedir)

    with open(filelist) as filelists:
        filelist = filelists.readlines()
    for i in range(len(filelist)):
        filelist[i] = filelist[i].strip("\n")
    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(seginfodir, i, savedir, dset) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for j in filelist:
            seg(seginfodir, j, savedir, dset)


# hand over parameter overview
# sys.argv[1] = sourcedir (str)
# sys.argv[2] = savedir (str)
# sys.argv[3] = filelistdir (str)
# sys.argv[4] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[5] = ifmulticore (str): If use multi processes.
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
