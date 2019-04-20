# Copyright 2017 Johns Hopkins University (Shinji Watanabe), Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json

import torch
from torch.utils.data import Dataset

from espnet.utils.training.trainer import Job


REPORT_INTERVAL = 100


class Evalution(Job):
    def __init__(self, model, loader, improve_hook, no_improve_hook, args):
        self.model = model
        self.loader = loader
        self.args = args
        self.best_acc = -float("inf")
        self.current_acc = 0.0
        self.no_improve = 0
        self.no_improve_hook = no_improve_hook

    def iterator(self):
        return self.loader

    def begin_epoch(self):
        self.model.eval()

    def process(self, data):
        with torch.no_grad():
            # TODO(karita) report
            result = self.model(data)
        return result

    def end_epoch(self):
        # TODO(karita) summary
        if self.current_acc > self.best_acc:
            self.best_acc = self.current_acc
        else:
            self.no_improve += 1
            self.no_improve_hook()

    def terminate(self):
        return self.args.patience < self.no_improve


class ASRDataset(Dataset):
    def __init__(self, json_path, args, subsampling_factor=0, preprocess_conf=None):
        from espnet.asr.asr_utils import make_batchset
        from espnet.utils.io_utils import LoadInputsAndTargets

        with open(json_path, 'rb') as f:
            self.utts = json.load(f)['utts']

        self.batchset = make_batchset(
            self.utts, args.batch_size,
            args.maxlen_in, args.maxlen_out, args.minibatches,
            min_batch_size=args.ngpu if args.ngpu > 1 else 1,
            shortest_first=args.sortagrad == -1 or args.sortagrad > 0)
        self.subsampling_factor = subsampling_factor
        self.load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=True, preprocess_conf=preprocess_conf)
        self.ignore_id = -1

    def __getitem__(self, index):
        from torch.nn.utils.rnn import pad_sequence

        xs, ys = self.load_inputs_and_targets(self.batchset[index])

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = torch.tensor([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_sequence([torch.tensor(x).float() for x in xs],
                              batch_first=True, padding_value=0)
        ys_pad = pad_sequence([torch.tensor(y).long() for y in ys],
                              batch_first=True, padding_value=self.ignore_id)
        return xs_pad, ilens, ys_pad

    def __len__(self):
        return len(self.batchset)


def train(args):
    # TODO(karita) copy required lines from asr.py
    pass
