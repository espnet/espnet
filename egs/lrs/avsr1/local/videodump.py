import os,sys
import json


def videodump(dumpavfile, dumpvideofile, dset):



    for root, dirs, files in os.walk(os.path.join(dumpavfile, dset)):
        for file in files:
            if '.json' in file:
                jsonname = file
                filename = os.path.join(root, file)
    with open(filename, encoding='utf-8') as json_file:
        avdata = json.load(json_file)
    delkey = []
    avkeys = avdata['utts'].keys()
    for avkey in avkeys:
        if '-' not in avkey:
            delkey.append(avkey)
        elif '-reverb' in avkey:
            delkey.append(avkey)
    for key in delkey:
        del avdata['utts'][key]


    savefilename = filename.replace(os.path.join(dumpavfile, dset), os.path.join(dumpvideofile, dset))
    if not os.path.exists(savefilename.replace(jsonname, '')):
        os.makedirs(savefilename.replace(jsonname, ''))
    with open(savefilename, 'w', encoding='utf-8') as f:
        json.dump(avdata, f, ensure_ascii=False, indent=4)

# hand over parameter overview
# sys.argv[1] = dumpfile (str), Directory to save audio-visual dump files
# sys.argv[2] = savedumpdir (str), Directory to save video dump files
# sys.argv[3] = dset (str), Which dataset


videodump(sys.argv[1],sys.argv[2],sys.argv[3])
