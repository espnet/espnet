import multiprocessing as mp
import os
import sys


def main(sourcedir, filelistdir, savedir, dset, nj, segment):
    """Prepare the Kaldi files.

    Args:
        sourcedir (str): LRS2 dataset dir.
        filelist (str): The dir of the mp4 file, it should be like
                        '5535415699068794046/00001'
        savedir (str): The dir save the Kaldi files.
        dset (str): Which set. For this code dset is pretrain set.
        nj (str): Number of multi processes.
        segment (str): If use segmentation

    """
    nj = int(nj)
    if nj > 1:
        multicore = True
    else:
        multicore = False
    if segment == "true":
        segment = True
    else:
        segment = False
    filelistdir = os.path.join(filelistdir, "Filelist_" + dset)
    with open(filelistdir) as filelists:
        filelist = filelists.readlines()
    for i in range(len(filelist)):
        filelist[i] = filelist[i].strip("\n")
    filelist.sort()
    if multicore is True:
        pool = mp.Pool(nj)
        job_args = [(i, dset, savedir, sourcedir, segment) for i in filelist]
        pool.map(product_helper, job_args)
    else:
        for i in filelist:
            set(i, dset, savedir, sourcedir, segment)


def product_helper(args):
    return set(*args)


def remove(sub, s):
    return s.replace(sub, "", -1)


def segmentation(textfiledir, file, segment=True):
    """Make segment information with segment interval 5s.

    Args:
        textfiledir (str): The Text and Segment File
        file (str): The file name

    """

    with open(textfiledir) as filelists:
        info = filelists.readlines()
    info[0] = remove("Text:  ", info[0])
    starttime = info[4].split(" ")[1]
    endtime = float(info[-1].split(" ")[2])
    segmentinfo = {file: {}}
    if segment is False:
        segmentinfo[file].update({str(0): {"segmenttime": [0.0, endtime]}})
        segmentinfo[file][str(0)].update({"segmenttext": info[0]})
        return segmentinfo
    else:
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

            cuttime = [starttime] + cuttime + [str(endtime)]
            # if endtime - float(cuttime[-2]) <= 1.0:
            #   del cuttime[-1]
            for cutid in range(len(cuttime) - 1):
                segmentinfo[file].update(
                    {str(cutid): {"segmenttime": [cuttime[cutid], cuttime[cutid + 1]]}}
                )

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
            for p in range(len(textlen)):
                temp = []
                for q in range(textlen[p]):
                    temp.append(text[q])
                segmentinfo[file][str(p)].update({"segmenttext": " ".join(temp)})
                text[0 : textlen[p]] = []

            return segmentinfo

        else:
            segmentinfo[file].update({str(0): {"segmenttime": [0.0, endtime]}})
            segmentinfo[file][str(0)].update({"segmenttext": info[0]})
            return segmentinfo


def set(file, s, savedir, sourcedir, segment):
    """Make the Kaldi files.

    Args:
        file (str): The file name.
        s (str): Which set. For this code dset is pretrain set.
        sourcedir: LRS2 dataset dir.
        savedir (str): The dir save the Kaldi files.

    """

    textdir = os.path.join(savedir, "text")
    utt2spkdir = os.path.join(savedir, "utt2spk")
    wavdir = os.path.join(savedir, "wav.scp")
    segmentdir = os.path.join(savedir, "segments")

    textfile = os.path.join(sourcedir, file + ".txt")
    mp4dir = os.path.join(sourcedir, file + ".mp4")

    if segment is False:
        segmentinfo = segmentation(textfile, file, segment)
        command1 = "ffmpeg -y -i"
        command2 = (
            "-vn -ac 1 -ar 16000 -ab 320k -f wav /tmp/tmp.$$; cat /tmp/tmp.$$ |\n"
        )
        splitname = file.split("/")
        Title = "_".join(["LRS2", splitname[0], splitname[1] + "p"])
        texttxt = [" ".join([Title, segmentinfo[file][str(0)]["segmenttext"]])]
        wavtxt = [" ".join([Title, command1, mp4dir, command2])]
        utttxt = [
            Title + " " + "_".join(["LRS2", splitname[0], splitname[1] + "p"]) + "\n"
        ]

    else:
        segdir = os.path.join(savedir, "seginfo.txt")
        segmentinfo = segmentation(textfile, file)
        if len(segmentinfo[file]) == 1:
            starttime = float(segmentinfo[file][str(0)]["segmenttime"][0])
            endtime = float(segmentinfo[file][str(0)]["segmenttime"][1])
            command1 = "ffmpeg -y -i"
            command2 = (
                "-vn -ac 1 -ar 16000 -ab 320k -f wav /tmp/tmp.$$; cat /tmp/tmp.$$ |\n"
            )
            splitname = file.split("/")
            Title = "_".join(
                [
                    "LRS2",
                    splitname[0],
                    splitname[1] + "p",
                    str(int(starttime * 100)).zfill(7),
                    str(int(endtime * 100)).zfill(7),
                ]
            )
            spkerid = "_".join(["LRS2", splitname[0], splitname[1] + "p"])
            texttxt = " ".join([Title, segmentinfo[file][str(0)]["segmenttext"]])
            wavtxt = " ".join([spkerid, command1, mp4dir, command2])
            segtxt = " ".join([Title, spkerid, str(starttime), str(endtime)]) + "\n"
            utttxt = Title + " " + spkerid + "\n"

        else:
            splitname = file.split("/")
            command1 = "ffmpeg -y -i"
            command2 = (
                "-vn -ac 1 -ar 16000 -ab 320k -f wav /tmp/tmp.$$; cat /tmp/tmp.$$ |\n"
            )
            spkerid = "_".join(["LRS2", splitname[0], splitname[1] + "p"])
            wavtxt = [" ".join([spkerid, command1, mp4dir, command2])]
            texttxt = []
            utttxt = []
            segtxt = []
            segmentinfos = []
            for i in range(len(segmentinfo[file])):
                starttime = float(segmentinfo[file][str(i)]["segmenttime"][0])
                endtime = float(segmentinfo[file][str(i)]["segmenttime"][1])
                if segmentinfo[file][str(i)]["segmenttext"] == "":
                    pass
                else:
                    Title = "_".join(
                        [
                            "LRS2",
                            splitname[0],
                            splitname[1] + "p",
                            str(int(starttime * 100)).zfill(7),
                            str(int(endtime * 100)).zfill(7),
                        ]
                    )
                    spkerid = "_".join(["LRS2", splitname[0], splitname[1] + "p"])
                    segtxt.append(
                        " ".join([Title, spkerid, str(starttime), str(endtime)]) + "\n"
                    )
                    segmentinfos.append(
                        " ".join([Title, mp4dir, str(starttime), str(endtime)]) + "\n"
                    )

                    temptext = " ".join(
                        [Title, segmentinfo[file][str(i)]["segmenttext"]]
                    )
                    if "\n" in temptext:
                        pass
                    else:
                        temptext = temptext + "\n"
                    texttxt.append(temptext)
                    utttxt.append(Title + " " + spkerid + "\n")

            with open(segdir, "a") as segprocess:
                segprocess.writelines(segmentinfos)
                segprocess.close()

    with open(textdir, "a") as textprocess:
        textprocess.writelines(texttxt)
        textprocess.close()
    with open(utt2spkdir, "a") as utt:
        utt.writelines(utttxt)
        utt.close()
    with open(wavdir, "a") as wav:
        wav.writelines(wavtxt)
        wav.close()
    with open(segmentdir, "a") as segs:
        segs.writelines(segtxt)
        segs.close()


# hand over parameter overview
# sys.argv[1] = sourcedir (str): The LRS2 dataset dir
#                                (e.g. /LRS2/data/lrs2_v1/mvlrs_v1/main)
# sys.argv[2] = filelistdir (str): The directory containing the dataset
#                                 Filelists (METADATA)
# sys.argv[3] = savedir (str): Save directory, datadir of the clean audio dataset
# sys.argv[4] = dset (str): Which set. For this code dset is pretrain set.
# sys.argv[5] = nj (str): Number of multi processes.
# sys.argv[6] = segment (str): If do segmentation.


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
