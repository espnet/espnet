import os,sys
import json
import multiprocessing as mp

def processing(i, audiodata, mfccdata, vdir, snrdir, vrmdir, dset):
    if 'noise' in i or 'reverb' in i:
        splitname = i.split('-')
        videoi = splitname[0]
    else:
        videoi = i
    if dset == 'pretrain':
        if 'LRS2' in i:
            vconfdir = os.path.join(vrmdir, 'LRS2' + dset, 'Conf')
            AUdir = os.path.join(vrmdir, 'LRS2' + dset, 'AUs')
        else:
            vconfdir = os.path.join(vrmdir, 'LRS3' + dset, 'Conf')
            AUdir = os.path.join(vrmdir, 'LRS3' + dset, 'AUs')
    else:
        vconfdir = os.path.join(vrmdir, 'LRS2' + dset, 'Conf')
        AUdir = os.path.join(vrmdir, 'LRS2' + dset, 'AUs')
    audiodata['utts'][i]['input'][0]['afeat'] = audiodata['utts'][i]['input'][0].pop('feat')
    audiodata['utts'][i]['input'][0]['ashape'] = audiodata['utts'][i]['input'][0].pop('shape')
    audiodata['utts'][i]['input'][0]['vfeat'] = vdir['input'][0]['feat']
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

def avpretraindump(dumpfile, dumpaudiofile, dumpvideofile, snrdir, vrmdir, mfccdumpdir, dset, ifmulticore):
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
    for root, dirs, files in os.walk(os.path.join(dumpvideofile, dset)):
        for file in files:
            if '.json' in file:
                videodumpfile = os.path.join(root, file)
    with open(videodumpfile, encoding='UTF-8') as videojson_file: # I think this should be dumpvideofile
        videodata = json.load(videojson_file)
    results = []
    if ifmulticore is True:
        keylist = list(audiodata['utts'].keys())
        pool = mp.Pool()
        job_args = [(i, audiodata, mfccdata, videodata['utts'][i.split('-')[0]], snrdir, vrmdir, dset) for i in keylist]
        results.extend(pool.map(product_helper, job_args))
    else:
        keylist = list(audiodata['utts'].keys())
        for i in keylist:
            results.append(processing(i, audiodata, mfccdata, videodata['utts'][i.split('-')[0]], snrdir, vrmdir, dset))

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
# sys.argv[3] = dumpvideofile (str), Directory to save video dump files
# sys.argv[4] = snrdir (str), Directory with SNR saved as .pt files
# sys.argv[5] = vrmdir (str), Directory with confidence saved as .pt files
# sys.argv[6] = mfccdumpdir (str), Directory to save mfcc dump files
# sys.argv[7] = dset(str), Which dataset
# sys.argv[8] = ifmulticore (boolean), If multi cpu processing should be used

avpretraindump(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
