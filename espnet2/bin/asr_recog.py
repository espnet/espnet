import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from typing import Sequence, Optional, Union, Dict

from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.train.dataset import Dataset, collate_fn
from espnet2.utils.device_funcs import to_device
from espnet2.utils.fileio import DatadirWriter


def recog(
        outdir: str,
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        dtype: str,
        beam_size: int,
        ngpu: int,
        char_list,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        nbest: int,
        num_workers: int,
        log_level: Union[int, str],
        path_and_name_and_type: Sequence[Sequence[str]],
        preprocess_conf: Dict[str, Union[str, dict]],
        asr_train_yaml: str,
        asr_model_file: str,
        lm_train_yaml: Optional[str],
        lm_model_file: Optional[str],
        word_lm_yaml: Optional[str],
        word_lm_model: Optional[str],
        ):
    if batch_size > 1:
        raise NotImplementedError('batch decoding is not implemented')
    if word_lm_yaml is not None:
        raise NotImplementedError('Word LM is not implemented')
    logging.basicConfig(
        level=log_level,
        format=
        '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    with Path(asr_train_yaml) as f:
        asr_train_args = yaml.load(f, Loader=yaml.Loader)
    asr_train_args = argparse.Namespace(**asr_train_args)
    asr_model = ASRTask.build_model(asr_train_args)
    asr_model.load_state_dict(asr_model_file)
    logging.info(f'ASR model: {asr_model}')

    encoder = asr_model.encoder
    decoder = asr_model.decoder
    ctc = asr_model.ctc

    scorers = dict(
        decoder=decoder,
        ctc=ctc,
        length_bonus=LengthBonus(len(char_list)),
    )

    if lm_train_yaml is not None:
        with Path(lm_train_yaml) as f:
            lm_train_args = yaml.load(f, Loader=yaml.Loader)
        lm_train_args = argparse.Namespace(**lm_train_args)
        lm = LMTask.build_model(lm_train_args)
        lm.load_state_dict(lm_model_file)
        scorers['lm'] = lm
        logging.info(f'LM model: {lm}')

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
        vocab_size=len(char_list),
        token_list=char_list,
    )
    logging.info(f'Beam_search: {beam_search}')

    if ngpu > 1:
        raise NotImplementedError('only single GPU decoding is supported')
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
        Dataset(path_and_name_and_type, preprocess_conf, float_dtype=dtype)
    logging.info(f'dataset: {dataset}')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers)
    keys_list = \
        [dataset.keys[i:i + batch_size]
         for i in range(
            0, int(np.ceil(len(dataset) / batch_size)), batch_size)]
    for keys, batch in zip(keys_list, loader):
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

        # FXIME(kamo): The output format should be discussed about
        with DatadirWriter(outdir) as writer:
            key = keys[0]
            for n, hyp in enumerate(nbest_hyps, 1):
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos and get results
                token_int = hyp.yseq[1:].tolist()
                # Change integer-ids to tokens
                token = [char_list[idx] for idx in token_int]

                # Create outdir / {n}best_recog
                ibest_writer = writer[f'{n}best_recog']

                # Write result to each files
                ibest_writer['token'][key] = ' '.join(token)
                ibest_writer['token_int'][key] = \
                    ' '.join(map(str, token_int))

                # Change integer-ids to tokens
                ibest_writer['text'][key] = \
                    ''.join(map(lambda x: x.replace('<space>', ' '), token))
                ibest_writer['score'][key] = str(hyp.score)
