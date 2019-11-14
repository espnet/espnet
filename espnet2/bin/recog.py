import logging

import torch

from espnet.asr.asr_mix_utils import add_results_to_json
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet2.tasks.asr import ASRTask
from espnet2.train.dataset import Dataset, BatchSampler, collate_fn
from espnet2.utils.device_funcs import to_device


def recog(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        dtype: str,
        beam_size: int,
        ngpu: int,
        char_list,
        ctc_weight: float,
        lm_weight: float,
        penalty: float):
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
    for scorer in scorers.items():
        if isinstance(scorer, torch.nn.Module):
            model.to(device=device, dtype=dtype).eval()

    # Creates eval-data-iterator
    data_conf = dict(input=input_file)
    dataset = Dataset(data_conf, preprocess, float_dtype=dtype)

    batch_sampler = BatchSampler(type='const', paths=eval_batch_files,
                                 batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for idx, keys in enumerate(batch_sampler):
            batch = collate_fn([dataset[k] for k in keys])
            batch = to_device(batch, device)
            enc = encoder(batch)
            nbest_hyps = beam_search(x=enc, maxlenratio=maxlenratio,
                                     minlenratio=minlenratio)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), nbest)]]
            new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))


def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score
