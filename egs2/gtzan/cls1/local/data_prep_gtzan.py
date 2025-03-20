import json
import os
import random
import sys

import soundfile as sf
from tqdm import tqdm

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]

def collect_fnames_for_each_genre(black_list): 

    # obtain genre tags 
    genres = sorted(os.listdir(DATA_READ_ROOT))
    
    # collect wavfile names for each genre
    fnames_for_each_genre = {}
    for genre in genres:
        fnames = os.listdir(os.path.join(DATA_READ_ROOT, genre))
        for fn in black_list: # remove fnames in black_list
            if fn in fnames:
                fnames.remove(fn)
                print('(data_prep_gtzan.py): removed file in black list '+fn)
            else:
                pass
        fnames_for_each_genre.update({genre:fnames})
    
    return fnames_for_each_genre


def split_fnames_with_balanced_genre(fnames_for_each_genre, ratio={'train':0.75, 'val':0.15, 'eval':0.10}):

    train_input, val_input, eval_input = [], [], [] # input
    train_label, val_label, eval_label = [], [], [] # labels

    # for each genre, randomize list and count files for train, val, and eval
    for genre in fnames_for_each_genre.keys():

        # randomized list 
        n_files = len(fnames_for_each_genre[genre])
        random_index = list(range(n_files))
        random.shuffle(random_index)

        # number of files for each split
        n_train = int(ratio['train'] * n_files)
        n_val = int(ratio['val'] * n_files)
        n_eval = n_files - (n_train + n_val)

        # store file names and labels
        train_input += [ fnames_for_each_genre[genre][i] for i in random_index[:n_train] ]
        train_label += [ genre for i in random_index[:n_train] ]
        val_input += [ fnames_for_each_genre[genre][i] for i in random_index[n_train:n_train+n_val] ]
        val_label += [ genre for i in random_index[n_train:n_train+n_val] ]
        eval_input += [ fnames_for_each_genre[genre][i] for i in random_index[-n_eval:] ]
        eval_label += [ genre for i in random_index[-n_eval:] ]        

    return train_input, train_label, val_input, val_label, eval_input, eval_label


# random.seed(0)
black_list = ['jazz.00054.wav'] # broken files
fnames_for_each_genre = collect_fnames_for_each_genre(black_list)

# data split 
# Split: 75% Train, 15% Val e 10% Test
ratio={'train':0.75, 'val':0.15, 'eval':0.10}
train_input, train_label, val_input, val_label, eval_input, eval_label = split_fnames_with_balanced_genre(fnames_for_each_genre, ratio)

# get shuffled indicies of data
train_set = list(range(len(train_input)))
val_set = list(range(len(val_input)))
eval_set = list(range(len(eval_input)))

random.shuffle(train_set)
random.shuffle(val_set)
random.shuffle(eval_set)


print(f"Train set size: {len(train_set)}")
print(f"Val set size: {len(val_set)}")
print(f"Eval set size: {len(eval_set)}")


"""
output kaldi style formats

text:
uttid000 rock
uttid025 pops
...

wav.scp:
uttid000 DATA_READ_ROOT/genre/fname 
...

utt2spk:
uttid000 rock
uttid025 pops
...
"""

for sets, inputs, labels, name in [(train_set, train_input, train_label, 'train'), (val_set, val_input, val_label, 'val'), (eval_set, eval_input, eval_label, 'eval')]:

    # check whether directory exists and create one if not
    os.makedirs(os.path.join(DATA_WRITE_ROOT, name), exist_ok=True)

    # paths to Kaldi style files
    text_write_path = os.path.join(DATA_WRITE_ROOT, name, "text")
    wav_scp_write_path = os.path.join(DATA_WRITE_ROOT, name, "wav.scp")
    utt2spk_write_path = os.path.join(DATA_WRITE_ROOT, name, "utt2spk")

    # output contents
    with open(text_write_path, "w") as text_f, open(wav_scp_write_path, "w") as wav_f, open(utt2spk_write_path, "w") as utt2spk_f:

        for i in sets:
            # text: 
            print(f"uttid{'{:0=4}'.format(i)} {labels[i]}", file=text_f)
            # wav.scp:
            print(f"uttid{'{:0=4}'.format(i)} {os.path.join(DATA_READ_ROOT, labels[i], inputs[i])}", file=wav_f)
            # utt2spk:
            print(f"uttid{'{:0=4}'.format(i)} {labels[i]}", file=utt2spk_f)
