import copy
import io
import json
import logging

import h5py
import kaldi_io_py
import numpy as np

from espnet.utils.processings.add_deltas import AddDeltas
from espnet.utils.processings.cmvn import CMVN
from espnet.utils.processings.spectrogram import LogMelSpectrogram
from espnet.utils.processings.spectrogram import Spectrogram
from espnet.utils.processings.spectrogram import Stft


def lazy_scp_reader(file_or_fd):
    fd = kaldi_io_py.open_or_fd(file_or_fd)
    key2mat = {}
    for line in fd:
        key, rxfile = line.decode().split(None, 1)
        key2mat[key] = rxfile

    def f(key):
        rxfile = key2mat[key]
        mat = kaldi_io_py.read_mat(rxfile)
        return mat
    return f


class PreProcessing(object):
    """Apply some functions to the mini-batch

    Examples:
        >>> kwargs = {"process": [{"type": "fbank",
        ...                        "n_mels": 80,
        ...                        "fs": 16000},
        ...                       {"type": "cmvn",
        ...                        "stats": "data/train/cmvn.ark",
        ...                        "norm_vars": True},
        ...                       {"type": "delta", "window": 2, "order": 2}],
        ...           "mode": "sequential"}
        >>> preprocessing = PreProcessing(**kwargs)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> processed_xs = preprocessing(xs)

    """

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            self.conf = {'mode': 'sequential', 'process': []}
        else:
            # Deep-copy to avoid sharing of mutable objects
            self.conf = copy.deepcopy(kwargs)
        self.cache = {}
        self.functions = {}
        self._config()

    def __repr__(self):
        rep = '\n' + '\n'.join(
            '{}: {}'.format(k, v) for k, v in self.functions.items()) + '\n'
        return '{}({})'.format(self.__class__.__name__, rep)

    def _config(self):
        if self.conf.get('mode', 'sequential') == 'sequential':
            for idx, process in enumerate(self.conf['process']):
                assert isinstance(process, dict), type(process)
                opts = dict(process)
                opts.pop('type')
                if process['type'] == 'fbank':
                    # x: array[Time]
                    self.functions[idx] = LogMelSpectrogram(**opts)

                elif process['type'] == 'spectrogram':
                    # x: array[Time]
                    self.functions[idx] = Spectrogram(**opts)

                elif process['type'] == 'stft':
                    # x: array[Channel, Time]
                    self.functions[idx] = Stft(**opts)

                elif process['type'] == 'delta':
                    self.functions[idx] = AddDeltas(**opts)

                elif process['type'] == 'cmvn':
                    self.functions[idx] = CMVN(**opts)

                elif process['type'] == 'wpe':
                    from espnet.utils.processings.wpe import WPE
                    # x: array[Channel, Time, Freq]
                    self.functions[idx] = WPE(**opts)
                else:
                    raise NotImplementedError(
                        'Not supporting: type={}'.format(process['type']))
        else:
            raise NotImplementedError(
                'Not supporting mode={}'.format(self.conf['mode']))

    def register(self, key, func):
        self.functions[key] = func

    def __call__(self, xs):
        """Return new mini-batch

        :param List[np.ndarray] xs:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        if self.conf.get('mode', 'sequential') == 'sequential':
            for idx in range(len(self.conf['process'])):
                func = self.functions[idx]
                try:
                    xs = [func(x) for x in xs]
                except Exception:
                    logging.fatal('Catch a exception from {}th func: {}'
                                  .format(idx, func))
                    raise
            return xs
        else:
            raise NotImplementedError(
                'Not supporting mode={}'.format(self.conf['mode']))


class LoadInputsAndTargets(object):
    def __init__(self, mode='asr',
                 preprocess_conf=None,
                 load_input=True,
                 load_output=True,
                 use_speaker_embedding=False,
                 use_second_target=False
                 ):
        self._loaders = {}
        if mode not in ['asr', 'tts']:
            raise ValueError(
                'Only asr or tts are allowed: mode={}'.format(mode))
        self.mode = mode
        if preprocess_conf is not None:
            with io.open(preprocess_conf, encoding='utf-8') as f:
                conf = json.load(f)
                assert isinstance(conf, dict), type(conf)
                self.preprocessing = PreProcessing(**conf)
            logging.warning(
                '[Experimental feature] Some pre-processings will be done '
                'for the mini-batch creation using {}'.format(preprocess_conf))
        else:
            # If conf doesn't exist, this function don't touch anything.
            self.preprocessing = None
        self.load_output = load_output
        self.load_input = load_input

        if use_second_target and use_speaker_embedding and mode == 'tts':
            raise ValueError('Choose one of "use_second_target" and '
                             '"use_speaker_embedding "')
        if (use_second_target or use_speaker_embedding) and mode != 'tts':
            logging.warning(
                '"use_second_target" and "use_speaker_embedding" is '
                'used only for tts mode')

        self.use_speaker_embedding = use_speaker_embedding
        self.use_second_target = use_second_target

    def __call__(self, batch):
        """Function to load inputs and targets from list of dicts

        :param List[Tuple[str, dict]] batch: list of dict which is subset of
            loaded data.json
        :return: list of input token id sequences [(L_1), (L_2), ..., (L_B)]
        :return: list of input feature sequences
            [(T_1, D), (T_2, D), ..., (T_B, D)]
        :rtype: list of float ndarray
        :return: list of target token id sequences [(L_1), (L_2), ..., (L_B)]
        :rtype: list of int ndarray
        """
        x_feats_list = []
        y_feats_list = []

        # Now "name" is never used, but getting them for the future extending
        x_names_list = []
        y_names_list = []

        for uttid, info in batch:
            x_feats = []
            y_feats = []
            x_names = []
            y_names = []

            keys = []
            if self.load_input:
                keys.append('input')
            if self.load_output:
                keys.append('output')

            for key in keys:
                for idx, inp in enumerate(info[key]):
                    if 'tokenid' in inp:
                        # ======= Legacy format for output =======
                        # {"output": [{"tokenid": "1 2 3 4"}])
                        assert isinstance(inp['tokenid'], str), \
                            type(inp['tokenid'])
                        x = np.fromiter(map(int, inp['tokenid'].split()),
                                        dtype=np.int64)
                    elif 'filetype' not in inp:
                        # ======= Legacy format for input =======
                        # {"input": [{"feat": "some/path.ark:123"}]),

                        if idx == 1 and self.mode == 'tts' \
                                and self.use_speaker_embedding:
                            x = kaldi_io_py.read_vec_flt(inp['feat'])
                        else:
                            x = kaldi_io_py.read_mat(inp['feat'])

                    else:
                        # ======= New format =======
                        # {"input":
                        #  [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
                        #    "filetype": "hdf5",

                        x = self._get_from_loader(
                            file_path=inp['feat'], loader_type=inp['filetype'])

                    if key == 'input':
                        x_feats.append(x)
                        x_names.append(inp['name'])

                    elif key == 'output':
                        y_feats.append(x)
                        y_names.append(inp['name'])

                # FIXME(kamo): Dirty way to load only speaker_embedding without the other inputs
                if not self.load_input and \
                        self.mode == 'tts' and self.use_speaker_embedding:
                    for idx, inp in enumerate(info['input']):
                        if idx != 1:
                            x = None
                        else:
                            if 'filetype' not in inp:
                                x = kaldi_io_py.read_vec_flt(inp['feat'])
                            else:
                                x = self._get_from_loader(
                                    file_path=inp['feat'],
                                    loader_type=inp['filetype'])
                        x_feats.append(x)
                        x_names.append(inp['name'])

            x_feats_list.append(x_feats)
            y_feats_list.append(y_feats)
            x_names_list.append(x_names)
            y_names_list.append(y_names)

        if self.mode == 'asr':
            return_batch = self._create_batch_asr(x_feats_list, y_feats_list)

        elif self.mode == 'tts':
            eos = int(batch[0][1]['output'][0]['shape'][1]) - 1
            return_batch = self._create_batch_tts(x_feats_list, y_feats_list,
                                                  eos)
        else:
            raise NotImplementedError

        if self.preprocessing is not None:
            if self.mode == 'asr':
                # Apply pre-processing only for the first item, now
                xs = return_batch[0]
                xs = self.preprocessing(xs)
                return (xs,) + return_batch[1:]

            elif self.mode == 'tts':
                xs = return_batch[1]
                xs = self.preprocessing(xs)
                return return_batch[0:1] + (xs,) + return_batch[2:]

            else:
                raise NotImplementedError
        else:
            return return_batch

    def _create_batch_asr(self, x_feats_list, y_feats_list):
        # Create a list from the first item
        xs = [x_list[0] for x_list in x_feats_list]

        # Assuming the names are common in the mini-batch
        if self.load_output:
            ys = [y_list[0] for y_list in y_feats_list]
            assert len(xs) == len(ys), (len(xs), len(ys))

            # get index of non-zero length samples
            nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(ys)))
        else:
            nonzero_idx = range(len(xs))

        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))

        if len(nonzero_sorted_idx) != len(xs):
            logging.warning(
                'Target sequences include empty tokenid (batch %d -> %d).' % (
                    len(xs), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [xs[i] for i in nonzero_sorted_idx]
        if self.load_output:
            ys = [ys[i] for i in nonzero_sorted_idx]
            return_batch = (xs, ys)
        else:
            return_batch = (xs,)
        return return_batch

    def _create_batch_tts(self, x_feats_list, y_feats_list, eos):
        # Use the output values as the input feats for tts mode
        xs = [y_list[0] for y_list in y_feats_list]
        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(xs[i]) > 0, range(len(xs)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
        # remove zero-length samples
        xs = [xs[i] for i in nonzero_sorted_idx]
        # Added eos into input sequence
        xs = [np.append(x, eos) for x in xs]

        if self.load_input:
            ys = [y_list[0] for y_list in x_feats_list]
            assert len(xs) == len(ys), (len(xs), len(ys))
            ys = [ys[i] for i in nonzero_sorted_idx]

            spembs = None
            spcs = None

            if self.use_second_target:
                spcs = [x_list[1] for x_list in x_feats_list]
                spcs = [spcs[i] for i in nonzero_sorted_idx]

            if self.use_speaker_embedding:
                spembs = [x_list[1] for x_list in x_feats_list]
                spembs = [spembs[i] for i in nonzero_sorted_idx]
            return_batch = (xs, ys, spembs, spcs)

        elif self.use_speaker_embedding:
            spembs = [x_list[1] for x_list in x_feats_list]
            spembs = [spembs[i] for i in nonzero_sorted_idx]
            return_batch = (xs, spembs)

        else:
            return_batch = (xs,)
        return return_batch

    def _get_from_loader(self, file_path, loader_type):
        """In order to make the fds to be opened only at the first referring,
        the loader are stored in self._loaders

        :param: str file_path
        :param: str loader_type
        :param: Hashable key
        :return:
        :rtype: np.ndarray

        """
        if loader_type in ['hdf5', 'h5']:
            file_path, key = file_path.split(':', 1)
            loader = self._loaders.get(file_path)
            if loader is None:
                #    {"input": [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
                #                "filetype": "hdf5",
                loader = h5py.File(file_path, 'r')
                self._loaders[file_path] = loader
            return loader[key].value
        elif loader_type == 'wav':
            raise NotImplementedError(
                'Not supported: loader_type={}'.format(loader_type))
        elif loader_type == 'npz':
            file_path, key = file_path.split(':', 1)
            loader = self._loaders.get(file_path)
            if loader is None:
                #    {"input": [{"feat": "some/path.npz:F01_050C0101_PED_REAL",
                #                "filetype": "npz",
                loader = np.load(file_path)
                self._loaders[file_path] = loader
            return loader[key]
        elif loader_type == 'npy':
            #    {"input": [{"feat": "some/path.npy",
            #                "filetype": "npy"},
            return np.load(file_path)
        elif loader_type == 'ark':
            #    {"input": [{"feat": "some/path.ark:123",
            #                "filetype": "ark"}]},
            return kaldi_io_py.read_mat(file_path)
        elif loader_type == 'vec':
            #    {"input": [{"feat": "some/path.ark:123",
            #                "filetype": "vec"}]},
            return kaldi_io_py.read_vec_flt(file_path)
        elif loader_type == 'scp':
            file_path, key = file_path.split(':', 1)
            loader = self._loaders.get(file_path)
            if loader is None:
                #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
                #                "filetype": "scp",
                loader = lazy_scp_reader(file_path)
            return loader[key]
        else:
            raise NotImplementedError(
                'Not supported: loader_type={}'.format(loader_type))
