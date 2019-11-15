import logging
from pathlib import Path

import torch

from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.train.dataset import Dataset, BatchSampler, collate_fn
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
        nbest: int):
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    model = ASRTask.build_model(asr_train_args)
    model.load_state_dict()
    lm = LMTask.build_model(lm_train_args)
    lm.load_state_dict()

    encoder = model.encoder
    decoder = model.decoder
    ctc = model.ctc

    scorers = dict(
        decoder=decoder,
        ctc=ctc,
        lm=lm,
        length_bonus=LengthBonus(len(char_list)),
    )

    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty)

    beam_search = BeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        vocab_size=len(char_list),
        token_list=char_list,
    )

    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"

    dtype = getattr(torch, dtype)

    logging.info(f"Decoding device={device}, dtype={dtype}")

    beam_search.to(device=device, dtype=dtype).eval()
    for scorer in scorers.values():
        if isinstance(scorer, torch.nn.Module):
            model.to(device=device, dtype=dtype).eval()

    # Creates eval-data-iterator
    data_conf = dict(input=input_file)
    dataset = Dataset(data_conf, preprocess, float_dtype=dtype)

    batch_sampler = BatchSampler(
        type='const', srcs=[input_file],
        batch_size=batch_size, shuffle=False)

    outpath = Path(outdir)

    for idx, keys in enumerate(batch_sampler):
        assert len(keys) == 1, len(keys)
        # 1. Create input data
        batch = collate_fn([dataset[k] for k in keys])
        batch = to_device(batch, device)
        key = keys[0]

        with torch.no_grad():
            # 2. Forward Encoder
            enc, _ = encoder(**batch)
            assert len(enc) == 1, len(enc)
            x = enc[0]

            # 3. Beam search
            nbest_hyps = beam_search(
                x=x, maxlenratio=maxlenratio, minlenratio=minlenratio)

        nbest_hyps = nbest_hyps[:nbest]

        # FXIME(kamo): The output format should be discussed about
        with DatadirWriter(outpath) as writer:
            for n, hyp in enumerate(nbest_hyps, 1):
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos and get results
                token_int = hyp.yseq[1:].tolist()
                # Change integer-ids to tokens
                token = [char_list[idx] for idx in token_int]

                # Create outdir / {n}best_recog{n}
                ibest_writer = writer[f'{n}best_recog']

                # Write result to each files
                ibest_writer['token'][key] = ' '.join(token)
                ibest_writer['token_int'][key] = \
                    ' '.join(map(str, token_int))

                # Change integer-ids to tokens
                ibest_writer['text'][key] = \
                    ''.join(map(lambda x: x.replace('<space>', ' '), token))
                ibest_writer['score'][key] = str(hyp.score)
