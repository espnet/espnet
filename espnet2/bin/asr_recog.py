import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Sequence, Optional, Union, Dict

import configargparse
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from typeguard import typechecked

from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.dataset import ESPNetDataset, our_collate_fn
from espnet2.utils.device_funcs import to_device
from espnet2.utils.fileio import DatadirWriter
from espnet2.utils.types import str2triple_str, NestedDictAction, str_or_none


@typechecked
def recog(
        output_dir: str,
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        dtype: str,
        beam_size: int,
        ngpu: int,
        seed: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        nbest: int,
        num_workers: int,
        log_level: Union[int, str],
        path_and_name_and_type: Sequence[Sequence[str]],
        key_file: Optional[str],
        preprocess: Optional[Dict[str, Union[str, dict]]],
        asr_train_config: str,
        asr_model_file: str,
        lm_train_config: Optional[str],
        lm_file: Optional[str],
        word_lm_train_config: Optional[str],
        word_lm_file: Optional[str],
        ):
    if batch_size > 1:
        raise NotImplementedError('batch decoding is not implemented')
    if word_lm_train_config is not None:
        raise NotImplementedError('Word LM is not implemented')
    if ngpu > 1:
        raise NotImplementedError('only single GPU decoding is supported')

    logging.basicConfig(
        level=log_level,
        format=
        '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    # Set random-seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    with Path(asr_train_config) as f:
        asr_train_args = yaml.load(f, Loader=yaml.Loader)
    asr_train_args = argparse.Namespace(**asr_train_args)
    asr_model = ASRTask.build_model(asr_train_args)
    asr_model.load_state_dict(torch.load(asr_model_file))
    logging.info(f'ASR model: {asr_model}')

    encoder = asr_model.encoder
    decoder = asr_model.decoder
    ctc = asr_model.ctc
    token_list = asr_model.token_list

    scorers = dict(
        decoder=decoder,
        ctc=ctc,
        length_bonus=LengthBonus(len(token_list)),
    )

    if lm_train_config is not None:
        with Path(lm_train_config) as f:
            lm_train_args = yaml.load(f, Loader=yaml.Loader)
        lm_train_args = argparse.Namespace(**lm_train_args)
        lm = LMTask.build_model(lm_train_args)
        lm.load_state_dict(torch.load(lm_file))
        scorers['lm'] = lm.lm
        logging.info(f'Language model: {lm}')

    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty)

    beam_search = BeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=asr_model.sos,
        eos=asr_model.eos,
        vocab_size=len(token_list),
        token_list=token_list,
    )
    logging.info(f'Beam_search: {beam_search}')

    if ngpu == 1:
        device = 'cuda'
    else:
        device = 'cpu'

    dtype = getattr(torch, dtype)
    logging.info(f'Decoding device={device}, dtype={dtype}')

    beam_search.to(device=device, dtype=dtype).eval()
    for scorer in scorers.values():
        if isinstance(scorer, torch.nn.Module):
            scorer.to(device=device, dtype=dtype).eval()

    # Creates eval-data-iterator
    dataset = \
        ESPNetDataset(path_and_name_and_type, preprocess, float_dtype=dtype)
    if key_file is None:
        key_file = path_and_name_and_type[0]

    batch_sampler = ConstantBatchSampler(
        batch_size=1, shape_file=key_file,
        shuffle=False, sort_in_batch=None)
    logging.info(f'dataset: {dataset}')
    loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler,
                        collate_fn=our_collate_fn, num_workers=num_workers)
    for keys, batch in zip(batch_sampler, loader):
        assert len(batch) == 1, len(batch)
        assert len(keys) == len(batch), f'Bug? {len(keys)} != {len(batch)}'
        with torch.no_grad():
            # 1. To device
            batch = to_device(batch, device)

            # 2. Forward Encoder
            enc, _ = encoder(**batch)
            assert len(enc) == 1, len(enc)

            # 3. Beam search
            nbest_hyps = beam_search(
                x=enc[0], maxlenratio=maxlenratio, minlenratio=minlenratio)
            nbest_hyps = nbest_hyps[:nbest]

        # FIXME(kamo): The output format should be discussed about
        with DatadirWriter(output_dir) as writer:
            key = keys[0]
            for n, hyp in enumerate(nbest_hyps, 1):
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos and get results
                token_int = hyp.yseq[1:].tolist()
                # Change integer-ids to tokens
                token = [token_list[idx] for idx in token_int]

                # Create outdir/{n}best_recog
                ibest_writer = writer[f'{n}best_recog']

                # Write result to each files
                ibest_writer['token'][key] = ' '.join(token)
                ibest_writer['token_int'][key] = ' '.join(map(str, token_int))

                # Change integer-ids to tokens
                ibest_writer['text'][key] = \
                    ''.join(map(lambda x: x.replace('<space>', ' '), token))
                ibest_writer['score'][key] = str(hyp.score)


def get_parser():
    parser = configargparse.ArgumentParser(
        description='ASR Decoding',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument(
        '--log_level', type=lambda x: str(x).upper(), default='INFO',
        choices=('INFO', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'),
        help='The verbose level of logging')

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ngpu', type=int, default=0,
                        help='The number of gpus. 0 indicates CPU mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument(
        '--dtype', default="float32",
        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
        help='Data type for training. '
             'O0,O1,.. flags require apex. '
             'See https://nvidia.github.io/apex/amp.html#opt-levels')
    parser.add_argument('--num_workers', type=int, default=1)

    group = parser.add_argument_group('Input data related')
    group.add_argument('--path_and_name_and_type', type=str2triple_str,
                       required=True, action='append')
    group.add_argument('--key_file', type=str_or_none)
    group.add_argument('--preprocess', type=NestedDictAction)

    group = parser.add_argument_group('The model configuration related')
    group.add_argument('--asr_train_config', type=str)
    group.add_argument('--asr_model_file', type=str)
    group.add_argument('--lm_train_config', type=str)
    group.add_argument('--lm_file', type=str)
    group.add_argument('--word_lm_train_config', type=str)
    group.add_argument('--word_lm_file', type=str)

    group = group.add_argument_group('Beam-search related')
    group.add_argument('--batch_size', type=int)
    group.add_argument('--nbest', type=int, default=1,
                       help='Output N-best hypotheses')
    group.add_argument('--beam_size', type=int, default=1,
                       help='Beam size')
    group.add_argument('--penalty', type=float, default=0.0,
                       help='Insertion penalty')
    group.add_argument('--maxlenratio', type=float, default=0.0,
                       help="Input length ratio to obtain max output length. "
                       "If maxlenratio=0.0 (default), it uses a end-detect "
                       "function "
                       "to automatically find maximum hypothesis lengths")
    group.add_argument('--minlenratio', type=float, default=0.0,
                       help='Input length ratio to obtain min output length')
    group.add_argument('--ctc_weight', type=float, default=0.0,
                       help='CTC weight in joint decoding')
    group.add_argument('--lm_weight', type=float, default=0.1,
                       help='RNNLM weight')

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    recog(**vars(args))


if __name__ == '__main__':
    main()
