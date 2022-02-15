import os,sys
import json
import multiprocessing as mp

def processing(i, audiodata):
    if audiodata['utts'][i]['input'][0]['shape'][0] < 10:
        return None
    else:
        audiodata['utts'][i]['input'][0]['afeat'] = audiodata['utts'][i]['input'][0].pop('feat')
        audiodata['utts'][i]['input'][0]['ashape'] = audiodata['utts'][i]['input'][0].pop('shape')


        return {i: audiodata['utts'][i]}

def product_helper(args):
    return processing(*args)

def audiojson(dumpfile, dumpsrcfile, dset, ifmulticore):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False

    output = {'utts': {}}
    for root, dirs, files in os.walk(os.path.join(dumpsrcfile, dset)):
        for file in files:
            if '.json' in file:
                jsonname = file
                filename = os.path.join(root, file)

    with open(filename, encoding='UTF-8') as json_file:
        audiodata = json.load(json_file)

    keylist = list(audiodata['utts'].keys())

    results = []
    if ifmulticore is True:
        pool = mp.Pool()
        job_args = [(i, audiodata) for i in keylist]
        results.extend(pool.map(product_helper, job_args))


    else:

        for i in keylist:
            results.append(processing(i, audiodata))

    for i in range(len(results)):
        if results[i] == None:
            pass
        else:
            output['utts'].update(results[i])


    savefilename = filename.replace(os.path.join(dumpsrcfile, dset), os.path.join(dumpfile, dset))
    if not os.path.exists(savefilename.replace(jsonname, '')):
        os.makedirs(savefilename.replace(jsonname, ''))
    with open(savefilename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)




# hand over parameter overview
# sys.argv[1] = dumpfile (str), Directory to save dump files
# sys.argv[2] = dumpsrcfile(str), Directory to save audio dump files
# sys.argv[3] = dset(str), Which dataset
# sys.argv[4] = ifmulticore (str), If multi cpu processing should be used

audiojson(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
