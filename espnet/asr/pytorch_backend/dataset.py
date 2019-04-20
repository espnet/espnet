# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from espnet.utils.io_utils import LoadInputsAndTargets


class ASRDataset(Dataset):
    '''ASR Dataset implementation

    :param List[Dict] batchset: you can create this by espnet.asr.asr_utils.make_batchset
    :param int subsampling_factor: input frame subsampling (skipping) factor.
    :param str preprocess_conf: file path for preprocess conf
    '''

    def __init__(self, batchset, subsampling_factor=0, preprocess_conf=None):
        self.batchset = batchset
        self.subsampling_factor = subsampling_factor
        self.load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=True, preprocess_conf=preprocess_conf)
        self.ignore_id = -1

    def __getitem__(self, index):

        xs, ys = self.load_inputs_and_targets(self.batchset[index])

        # TODO(karita) make this subsampling inside model
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
