import os,sys
import json
import torch
import multiprocessing as mp

def processing(i, audiodata, mfccdata, vdir, videodir, snrdir, vrmdir, dset, videonoise):
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
        if videonoise == 'None':
            vconfdir = os.path.join(vrmdir, 'LRS2' + dset, 'Conf')
            AUdir = os.path.join(vrmdir, 'LRS2' + dset, 'AUs')
            videofeatdir = os.path.join(vrmdir, 'LRS2' + dset, 'Pics')
        else:
            vconfdir = os.path.join(vrmdir, 'LRS2' + dset + '_' + videonoise, 'Conf')
            AUdir = os.path.join(vrmdir, 'LRS2' + dset + '_' + videonoise, 'AUs')
            videofeatdir = os.path.join(vrmdir, 'LRS2' + dset + '_' + videonoise, 'Pics')
    audiodata['utts'][i]['input'][0]['afeat'] = audiodata['utts'][i]['input'][0].pop('feat')
    audiodata['utts'][i]['input'][0]['ashape'] = audiodata['utts'][i]['input'][0].pop('shape')
    audiodata['utts'][i]['input'][0]['vfeat'] = os.path.join(videofeatdir, videoi + ".pt")
    videodata = torch.load(os.path.join(videofeatdir, videoi + ".pt"))
    audiodata['utts'][i]['input'][0]['vshape'] = [videodata.shape[0], videodata.shape[1], videodata.shape[2]]
    audiodata['utts'][i]['input'][0]['vshape'] = vdir['input'][0]['shape']
    audiodata['utts'][i]['input'][0]['aRMs'] = os.path.join(snrdir, i + ".pt")
    audiodata['utts'][i]['input'][0]['vRMs'] = os.path.join(vconfdir, videoi + ".pt")
    audiodata['utts'][i]['input'][0]['aRMshape'] = [audiodata['utts'][i]['input'][0]['ashape'][0], 1]
    audiodata['utts'][i]['input'][0]['vRMshape'] = [audiodata['utts'][i]['input'][0]['vshape'][0], 1]
    audiodata['utts'][i]['input'][0]['mfcc'] = mfccdata['utts'][i]['input'][0]['feat']
    audiodata['utts'][i]['input'][0]['mfccshape'] = mfccdata['utts'][i]['input'][0]['shape']
    audiodata['utts'][i]['input'][0]['AUs'] = os.path.join(AUdir, videoi + ".pt")
    audiodata['utts'][i]['input'][0]['AUshape'] = [audiodata['utts'][i]['input'][0]['vshape'][0], 6]

    return {i: audiodata['utts'][i]}


def product_helper(args):
    return processing(*args)

def avtraindecodedump(dumpfile, dumpaudiofile, dumpvideofile, videodir, snrdir, vrmdir, mfccdumpdir, dset, noisecombination, ifmulticore):
    audionoise = noisecombination.split('_')[0]
    videonoise = noisecombination.split('_')[1]
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False

    snrdir = os.path.join(snrdir, dset + '_' + audionoise)
    output = {'utts': {}}
    if videonoise == 'None':
        videodset = dset
    else:
        videodset = dset + '_decode_' + videonoise
    for root, dirs, files in os.walk(os.path.join(dumpaudiofile, dset + '_decode_' + audionoise)):
        for file in files:
            if '.json' in file:
                jsonname = file
                filename = os.path.join(root, file)
    with open(filename, encoding='UTF-8') as json_file:
        audiodata = json.load(json_file)
    for root, dirs, files in os.walk(os.path.join(mfccdumpdir, dset + '_decode_' + audionoise)):
        for file in files:
            if '.json' in file:
                mfccdumpfile = os.path.join(root, file)
    with open(mfccdumpfile, encoding='UTF-8') as mfccjson_file:
        mfccdata = json.load(mfccjson_file)
    for root, dirs, files in os.walk(os.path.join(dumpvideofile, videodset)):
        for file in files:
            if '.json' in file:
                videodumpfile = os.path.join(root, file)
    with open(videodumpfile, encoding='UTF-8') as mfccjson_file:
        videodata = json.load(mfccjson_file)
    results = []
    if ifmulticore is True:
        keylist = list(audiodata['utts'].keys())
        pool = mp.Pool()
        job_args = [(i, audiodata, mfccdata, videodata['utts'][i.split('-')[0]], videodir, snrdir, vrmdir, dset, videonoise) for i in keylist]
        results.extend(pool.map(product_helper, job_args))
    else:
        keylist = list(audiodata['utts'].keys())
        results = []
        for i in keylist:
            results.append(processing(i, audiodata, mfccdata, videodata['utts'][i.split('-')[0]], videodir, snrdir, vrmdir, dset, videonoise))

    for i in range(len(results)):
        output['utts'].update(results[i])
    if videonoise == 'None':
        savename = dset + '_' + audionoise
    else:
        savename = dset + '_' + videonoise
    savefilename = filename.replace(os.path.join(dumpaudiofile, dset + '_decode_' + audionoise), os.path.join(dumpfile, savename))
    if not os.path.exists(savefilename.replace(jsonname, '')):
        os.makedirs(savefilename.replace(jsonname, ''))
    with open(savefilename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

# hand over parameter overview
# sys.argv[1] = dumpfile (str), Directory to save dump files
# sys.argv[2] = dumpaudiofile (str), Directory to save audio dump files
# sys.argv[3] = dumpvideofile (str), Directory to save video dump files
# sys.argv[4] = videodir (str), Directory to save video dump files
# sys.argv[5] = snrdir (str), Directory with SNR saved as .pt files
# sys.argv[6] = vrmdir (str), Directory with confidence saved as .pt files
# sys.argv[7] = mfccdumpdir (str), Directory to save mfcc dump files
# sys.argv[8] = dset (str), Which dataset
# sys.argv[9] = noisecombination (str), Augumented audio and video noise type
# sys.argv[10] = ifmulticore (boolean), If multi cpu processing should be used


avtraindecodedump(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6],sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
