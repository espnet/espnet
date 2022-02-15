import os,sys
import json
import multiprocessing as mp

def processing(i, audiodata, mfccdata, vdir, snrdir, vconfdir, AUdir):
    if 'noise' in i or 'reverb' in i:
        audioi = i
        i = i.split('-')
        videoi = i[0]
    else:
        audioi = i
        videoi = i
    audiodata['utts'][audioi]['input'][0]['afeat'] = audiodata['utts'][audioi]['input'][0].pop('feat')
    audiodata['utts'][audioi]['input'][0]['ashape'] = audiodata['utts'][audioi]['input'][0].pop('shape')
    audiodata['utts'][audioi]['input'][0]['vfeat'] = os.path.join(vdir, videoi + ".pt")
    audiodata['utts'][audioi]['input'][0]['aRMs'] = os.path.join(snrdir, audioi + ".pt")
    audiodata['utts'][audioi]['input'][0]['vRMs'] = os.path.join(vconfdir, videoi + ".pt")
    audiodata['utts'][audioi]['input'][0]['mfcc'] = mfccdata['utts'][audioi]['input'][0]['feat']
    audiodata['utts'][audioi]['input'][0]['AUs'] = os.path.join(AUdir, videoi + ".pt")

    return {audioi: audiodata['utts'][audioi]}


def product_helper(args):
    return processing(*args)

def remakejson(dumpfile, dumpavfile, dumpvideofile, vdir, snrdir, vconfdir, mfccdumpfile, AUdir, ifmulticore):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False

    output = {'utts': {}}
    if ifmulticore is True:
        global audiodata
        with open(dumpfile) as json_file:
            audiodata = json.load(json_file)
        with open(mfccdumpfile) as mfccjson_file:
            mfccdata = json.load(mfccjson_file)
        keylist = list(audiodata['utts'].keys())
        pool = mp.Pool()
        job_args = [(i, audiodata, mfccdata, vdir, snrdir, vconfdir, AUdir) for i in keylist]
        results = pool.map(product_helper, job_args)
    else:
        with open(dumpfile) as json_file:
            audiodata = json.load(json_file)
        with open(mfccdumpfile) as mfccjson_file:
            mfccdata = json.load(mfccjson_file)
        keylist = list(audiodata['utts'].keys())
        results = []
        for i in keylist:
            results.append(processing(i, audiodata, mfccdata, vdir, snrdir, vconfdir, AUdir))

    for i in range(len(results)):
        output['utts'].update(results[i])

    keylist = list(output['utts'].keys())
    keydict = {}
    keysublist = []
    for key in keylist:
        key = key.strip("-noise")
        key = key.strip("-reverb")
        keysublist.append(key)
    keysublist = list(set(keysublist))
    [keydict.update({i: []}) for i in keysublist]
    for key in keylist:
        if "-" in key:
            keydict[key.split("-")[0]].append(key)
        else:
            keydict[key].append(key)
    outvideodict = {}
    outvideodict.update({'utts': {}})
    for i in keydict.keys():
        outvideodict['utts'].update({keydict[i][0]: output['utts'][keydict[i][0]]})

    with open(dumpvideofile, 'w', encoding='utf-8') as f:
        json.dump(outvideodict, f, ensure_ascii=False, indent=4)


    with open(dumpavfile, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

# hand over parameter overview
# sys.argv[1] = dumpfile (str), Directory to save dump files
# sys.argv[2] = dumpavfile (str), Directory to save audio-visual dump files
# sys.argv[3] = dumpvideofile (str), Directory to save video dump files
# sys.argv[4] = vdir (str), Video feature directory
# sys.argv[5] = snrdir (str), Directory with SNR saved as .pt files
# sys.argv[6] = vconfdir (str), Directory with confidence saved as .pt files
# sys.argv[7] = mfccdumpfile (str), Directory to save mfcc dump files
# sys.argv[8] = AUdir (str), Directory with Facial Action Units saved as .pt files
# sys.argv[9] = ifmulticore (boolean), If multi cpu processing should be used

remakejson(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6],sys.argv[7], sys.argv[8], sys.argv[9])
