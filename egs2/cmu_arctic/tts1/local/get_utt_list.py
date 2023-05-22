import glob
import os

import numpy as np

# This file creates train-dev-eval utterance list for all arctic voices.
# dev split includes arctic_b0340 - arctic_b439
# eval split includes arctic_b0440 - larger
utt_train_list = "data/all/utt_train_list.txt"
utt_dev_list = "data/all/utt_dev_list.txt"
utt_eval_list = "data/all/utt_eval_list.txt"

ftrain = open(utt_train_list, "w")
fdev = open(utt_dev_list, "w")
feval = open(utt_eval_list, "w")
for spk in ["slt", "clb", "bdl", "rms", "jmk", "awb", "ksp"]:
    for w in glob.glob("downloads/cmu_us_all_arctic/wav/{}_*.wav".format(spk, spk)):
        # check if in train list
        file_suffix = os.path.basename(w).split("_")[2]

        if file_suffix.startswith("a"):
            ftrain.write("{}\n".format(os.path.basename(w)[:-4]))
        else:
            # filename starts with b, check the number
            filenum = int(file_suffix[1:-4])
            if filenum <= 339:
                ftrain.write("{}\n".format(os.path.basename(w)[:-4]))
            elif filenum <= 439:
                fdev.write("{}\n".format(os.path.basename(w)[:-4]))
            else:
                feval.write("{}\n".format(os.path.basename(w)[:-4]))

ftrain.close()
fdev.close()
feval.close()
