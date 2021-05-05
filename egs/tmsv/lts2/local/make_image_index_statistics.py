import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

spk_dict = {}
utt_dict = {}
num_of_vector = 128
with open('data/train/text', 'r') as f:
    for line in f.readlines():
        print(line)
        line = line.strip()
        ls = line.split(' ')
        name = ls[0]
        spk = name.split('_')[0]
        utt = name.split('_')[1]

        if spk not in spk_dict.keys():
            spk_dict[spk] = [0 for i in range(num_of_vector)]
        if utt not in utt_dict.keys():
            utt_dict[utt] = [0 for i in range(num_of_vector)]
        for i in range(1, len(ls)):
            spk_dict[spk][int(ls[i])-1] += 1
            utt_dict[utt][int(ls[i])-1] += 1

    spk_dir = 'image_statistics_cd{}/spk'.format(num_of_vector)
    utt_dir = 'image_statistics_cd{}/utt'.format(num_of_vector)

    os.makedirs(spk_dir)
    os.makedirs(utt_dir)

    for spk in spk_dict.keys():
        print(spk)
        x = np.arange(num_of_vector)
        plt.bar(x, spk_dict[spk])
        plt.savefig(os.path.join(spk_dir, '{}.jpg'.format(spk)))
        plt.close()
    for utt in utt_dict.keys():
        print(utt)
        x = np.arange(num_of_vector)
        plt.bar(x, utt_dict[utt])
        plt.savefig(os.path.join(utt_dir, '{}.jpg'.format(utt)))
        plt.close()


        

