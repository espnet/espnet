import os,sys
import json
import multiprocessing as mp

def processing(i, audiodata):
    audiodata['utts'][i]['input'][0]['afeat'] = audiodata['utts'][i]['input'][0].pop('feat')
    audiodata['utts'][i]['input'][0]['ashape'] = audiodata['utts'][i]['input'][0].pop('shape')


    return {i: audiodata['utts'][i]}

def product_helper(args):
    return processing(*args)

def extractsnr(srcdir, noisetype):
    if not os.path.exists(os.path.join(srcdir, noisetype)):
        os.makedirs(os.path.join(srcdir, noisetype))
    with open(os.path.join(srcdir, 'Testfbank_aug_' + noisetype, 'wav.scp'), "r") as l:
        wav = l.readlines()
        l.close()
    snrdict = {}
    snrlist = ['-12', '-9', '-6', '-3', '0', '3',  '6', '9', '12']
    for snr in snrlist:
        snrdict.update({snr: []})
    for i in wav:
        splittext = i.split(' ')
        for j in splittext:
            if 'snrs' in j:
                snrinfo = j
        snrdata = snrinfo.split('\'')[1]
        if ',' in snrdata:
            snrsplit = snrdata.split(',')
            snrsplit = [int(x) for x in snrsplit]
            snrdata = str(min(snrsplit))
        snrdict[str(snrdata)].append(splittext[0] + ' ' + str(snrdata) + '\n')

    for snr in snrlist:
        with open(os.path.join(srcdir, noisetype, 'Test_' + snr), 'w') as f:
            f.writelines(snrdict[snr])



# hand over parameter overview
# sys.argv[1] = srcdir (str), Source directoy
# sys.argv[2] = noisetype(str), Noise type


extractsnr(sys.argv[1], sys.argv[2])
