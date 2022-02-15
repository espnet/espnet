import os,sys
import json
import torch
import multiprocessing as mp

def processing(i, audiodata, mfccdata, videodir, snrdir, vrmdir, dset):
    if 'noise' in i or 'reverb' in i:
        splitname = i.split('-')
        videoi = splitname[0]
    else:
        videoi = i
    if dset == 'pretrain':
        if 'LRS2' in i:
            vconfdir = os.path.join(vrmdir, 'LRS2' + dset, 'Conf')
            AUdir = os.path.join(vrmdir, 'LRS2' + dset, 'AUs')
            videofeatdir = os.path.join(videodir, 'LRS2' + dset)
        else:
            vconfdir = os.path.join(vrmdir, 'LRS3' + dset, 'Conf')
            AUdir = os.path.join(vrmdir, 'LRS3' + dset, 'AUs')
            videofeatdir = os.path.join(videodir, 'LRS3' + dset)
    else:
        vconfdir = os.path.join(vrmdir, 'LRS2' + dset, 'Conf')
        AUdir = os.path.join(vrmdir, 'LRS2' + dset, 'AUs')
        videofeatdir = os.path.join(vrmdir, 'LRS2' + dset, 'Pics')
    audiodata['utts'][i]['input'][0]['afeat'] = audiodata['utts'][i]['input'][0].pop('feat')
    audiodata['utts'][i]['input'][0]['ashape'] = audiodata['utts'][i]['input'][0].pop('shape')
    audiodata['utts'][i]['input'][0]['vfeat'] = os.path.join(videofeatdir, videoi + ".pt")
    videodata = torch.load(os.path.join(videofeatdir, videoi + ".pt"))
    audiodata['utts'][i]['input'][0]['vshape'] = [videodata.shape[0], videodata.shape[1], videodata.shape[2]]
    audiodata['utts'][i]['input'][0]['aRMs'] = os.path.join(snrdir, i + ".pt")
    audiodata['utts'][i]['input'][0]['vRMs'] = os.path.join(vconfdir, videoi + ".pt")
    audiodata['utts'][i]['input'][0]['aRMshape'] = [audiodata['utts'][i]['input'][0]['ashape'][0], 1]
    audiodata['utts'][i]['input'][0]['mfcc'] = mfccdata['utts'][i]['input'][0]['feat']
    audiodata['utts'][i]['input'][0]['mfccshape'] = mfccdata['utts'][i]['input'][0]['shape']
    audiodata['utts'][i]['input'][0]['AUs'] = os.path.join(AUdir, videoi + ".pt")

    return {i: audiodata['utts'][i]}


def product_helper(args):
    return processing(*args)

def avpretraindump(dumpfile, dumpaudiofile, videodir, snrdir, vrmdir, mfccdumpdir, dset, ifmulticore):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False

    snrdir = os.path.join(snrdir, dset)
    output = {'utts': {}}
    for root, dirs, files in os.walk(os.path.join(dumpaudiofile, dset)):
        for file in files:
            if '.json' in file:
                jsonname = file
                filename = os.path.join(root, file)
    with open(filename, encoding='UTF-8') as json_file:
        audiodata = json.load(json_file)
    for root, dirs, files in os.walk(os.path.join(mfccdumpdir, dset)):
        for file in files:
            if '.json' in file:
                mfccdumpfile = os.path.join(root, file)
    with open(mfccdumpfile, encoding='UTF-8') as mfccjson_file:
        mfccdata = json.load(mfccjson_file)
    results = []
    if ifmulticore is True:
        keylist = list(audiodata['utts'].keys())
        pool = mp.Pool()
        job_args = [(i, audiodata, mfccdata, videodir, snrdir, vrmdir, dset) for i in keylist]
        results.extend(pool.map(product_helper, job_args))
    else:
        keylist = list(audiodata['utts'].keys())
        for i in keylist:
            results.append(processing(i, audiodata, mfccdata, videodir, snrdir, vrmdir, dset))

    for i in range(len(results)):
        output['utts'].update(results[i])

    savefilename = filename.replace(os.path.join(dumpaudiofile, dset), os.path.join(dumpfile, dset))
    if not os.path.exists(savefilename.replace(jsonname, '')):
        os.makedirs(savefilename.replace(jsonname, ''))
    with open(savefilename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

# hand over parameter overview
# sys.argv[1] = dumpfile (str), Directory to save dump files
# sys.argv[2] = dumpaudiofile (str), Directory to save audio dump files
# sys.argv[3] = videodir (str), Directory to save video feature files
# sys.argv[4] = snrdir (str), Directory with SNR saved as .pt files
# sys.argv[5] = vrmdir (str), Directory with confidence saved as .pt files
# sys.argv[6] = mfccdumpdir (str), Directory to save mfcc dump files
# sys.argv[7] = dset(str), Which dataset
# sys.argv[8] = ifmulticore (boolean), If multi cpu processing should be used


avpretraindump(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6],sys.argv[7], sys.argv[8])
