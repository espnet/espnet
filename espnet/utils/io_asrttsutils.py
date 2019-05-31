from collections import OrderedDict
import io
import logging
import os

import h5py
import kaldiio
import numpy as np
import soundfile

from espnet.transform.transformation import Transformation


class LoadInputsAndTargetsASRTTS(object):
    """Create a mini-batch from a list of dicts

    >>> batch = [('utt1',
    ...           dict(input=[dict(feat='some.ark:123',
    ...                            filetype='mat',
    ...                            name='input1',
    ...                            shape=[100, 80])],
    ...                output=[dict(tokenid='1 2 3 4',
    ...                             name='target1',
    ...                             shape=[4, 31])]]))
    >>> l = LoadInputsAndTargets()
    >>> feat, target = l(batch)

    :param: str mode: Specify the task mode, "asr" or "tts"
    :param: str preprocess_conf: The path of a json file for pre-processing
    :param: bool load_input: If False, not to load the input data
    :param: bool load_output: If False, not to load the output data
    :param: bool sort_in_input_length: Sort the mini-batch in descending order
        of the input length
    :param: bool use_speaker_embedding: Used for asrtts mode only
    :param: bool use_second_target: Used for tts mode only
    :param: dict preprocess_args: Set some optional arguments for preprocessing
    :param: Optional[dict] preprocess_args: Used for tts mode only
    """

    def __init__(self, mode='asr',
                 preprocess_conf=None,
                 load_input=True,
                 load_output=True,
                 sort_in_input_length=True,
                 use_speaker_embedding=False,
                 use_second_target=False,
                 preprocess_args=None
                 ):
        self._loaders = {}
        if mode not in ['asr', 'tts']:
            raise ValueError(
                'Only asr or tts are allowed: mode={}'.format(mode))
        if preprocess_conf is not None:
            self.preprocessing = Transformation(preprocess_conf)
            logging.warning(
                '[Experimental feature] Some preprocessing will be done '
                'for the mini-batch creation using {}'
                .format(self.preprocessing))
        else:
            # If conf doesn't exist, this function don't touch anything.
            self.preprocessing = None

        if use_second_target and use_speaker_embedding and mode == 'tts':
            raise ValueError('Choose one of "use_second_target" and '
                             '"use_speaker_embedding "')
        if (use_second_target or use_speaker_embedding) and mode != 'tts':
            logging.warning(
                '"use_second_target" and "use_speaker_embedding" is '
                'used only for tts mode')

        self.mode = mode
        self.load_output = load_output
        self.load_input = load_input
        self.sort_in_input_length = sort_in_input_length
        self.use_speaker_embedding = use_speaker_embedding
        self.use_second_target = use_second_target
        if preprocess_args is None:
            self.preprocess_args = {}
        else:
            assert isinstance(preprocess_args, dict), type(preprocess_args)
            self.preprocess_args = dict(preprocess_args)

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
        x_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
        y_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
        uttid_list = []  # List[str]

        for uttid, info in batch:
            uttid_list.append(uttid)

            if self.load_input:
                # Note(kamo): This for-loop is for multiple inputs
                for idx, inp in enumerate(info['input']):
                    x = self._get_from_loader(filepath=inp['feat'], filetype=inp.get('filetype', 'mat'))
                    x_feats_dict.setdefault(inp['name'], []).append(x)
            # FIXME(kamo): Dirty way to load only speaker_embedding without the other inputs
            elif self.mode == 'tts' and self.use_speaker_embedding:
                for idx, inp in enumerate(info['input']):
                    if idx != 1:
                        x = None
                    else:
                        x = self._get_from_loader(
                            filepath=inp['feat'],
                            filetype=inp.get('filetype', 'mat'))
                    x_feats_dict.setdefault(inp['name'], []).append(x)

            if self.load_output:
                for idx, inp in enumerate(info['output']):
                    if 'tokenid' in inp:
