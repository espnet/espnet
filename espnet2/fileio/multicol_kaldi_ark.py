import kaldiio
import numpy as np

class MultiColKaldiArkReader:
    def __init__(self, path):
        self.utt2example = {}
        for line in open(path):
            elems = line.strip().split()
            name = elems[0]
            contents = elems[1:]
            self.utt2example[name] = contents

    def __getitem__(self, name):
        # load all items and concat. previously we randomly choose one in the DPO work
        retval = np.concatenate([
            kaldiio.load_mat(content) for content in self.utt2example[name]
        ])
        return retval

    def __len__(self):
        return len(self.utt2example)
