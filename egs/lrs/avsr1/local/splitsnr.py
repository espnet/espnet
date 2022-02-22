import copy
import json
import os
import sys


def checkkeylist(keylist, vklist):
    vdict = {}
    keylistnew = copy.deepcopy(keylist)
    for i in range(len(keylist)):
        keylist[i] = keylist[i].strip("-noise")
        keylist[i] = keylist[i].strip("-reverb")
    for i in range(len(vklist)):
        temp = vklist[i]
        vklist[i] = vklist[i].split("_")
        vklist[i] = "_".join((vklist[i][1], vklist[i][2][:-1]))
        vdict.update({vklist[i]: temp})
    keylist = list(set(keylist))
    missing = set(keylist).difference(vklist)
    for i in missing:
        keylistnew = [x for x in keylistnew if i not in x]
    return keylistnew, vdict


def checktrainkeylist(keylist, vklist):
    keylistnew = copy.deepcopy(keylist)
    for i in range(len(keylist)):
        keylist[i] = keylist[i].strip("-noise")
        keylist[i] = keylist[i].strip("-reverb")

    keylist = list(set(keylist))
    missing = set(keylist).difference(vklist)
    for i in missing:
        keylistnew = [x for x in keylistnew if i not in x]
    return keylistnew


def splitsnr(srcdir, noisecombination, snrdir):
    audionoise = noisecombination.split("_")[0]
    videonoise = noisecombination.split("_")[1]
    if videonoise == "None":
        dumpfile = "Test_" + audionoise
    else:
        dumpfile = "Test_" + videonoise
    if "music" not in noisecombination:
        noisetype = "noise"
    else:
        noisetype = "music"
    snrsdir = os.path.join(snrdir, noisetype)

    dumpdir = os.path.join(srcdir, dumpfile)
    for root, dirs, files in os.walk(dumpdir):
        for file in files:
            if ".json" in file:
                jsonname = file
                dumpfile = os.path.join(root, file)
    with open(dumpfile, encoding="UTF-8") as json_file:
        data = json.load(json_file)

    outdict = {}
    snrlist = ["-12", "-9", "-6", "-3", "0", "3", "6", "9", "12"]
    for snr in snrlist:
        outdict.update({snr: {"utts": {}}})
    outdict.update({"clean": {"utts": {}}})
    outdict.update({"reverb": {"utts": {}}})
    uttskey = data["utts"].keys()
    for utts in uttskey:
        if "-reverb" in utts:
            outdict["reverb"]["utts"].update({utts: data["utts"][utts]})
        if "-" not in utts:
            outdict["clean"]["utts"].update({utts: data["utts"][utts]})
    for snr in snrlist:
        snrfiledir = os.path.join(snrsdir, "Test_" + snr)
        with open(snrfiledir, "r") as f:
            snrfilelist = f.readlines()
            f.close()
        for j in snrfilelist:
            splitname = j.split(" ")[0]
            outdict[snr]["utts"].update({splitname: data["utts"][splitname]})

    snrlist.extend(["clean", "reverb"])
    for snr in snrlist:
        savefilename = dumpfile.replace(dumpdir, os.path.join(dumpdir, snr))
        if not os.path.exists(savefilename.replace(jsonname, "")):
            os.makedirs(savefilename.replace(jsonname, ""))
        with open(savefilename, "w", encoding="utf-8") as f:
            json.dump(outdict[snr], f, ensure_ascii=False, indent=4)


# hand over parameter overview
# sys.argv[1] = srcdir (str), Source directoy
# sys.argv[2] = noisecombination(str), Noise combination
#               (noise_None' 'music_None' 'noise_blur' 'noise_saltandpepper)
# sys.argv[3] = snrdir(str)


splitsnr(sys.argv[1], sys.argv[2], sys.argv[3])
