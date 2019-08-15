from argparse import Namespace
import torch

from espnet.nets.pytorch_backend.beam_search import beam_search
from espnet.nets.pytorch_backend.beam_search import LengthBonus
from espnet.nets.pytorch_backend.lm.legacy import LegacyRNNLM

from test.test_e2e_asr_transformer import prepare


def test_beam_search_equal():
    ctc = 0.5                   # TODO(karita) non-zero
    model, x, ilens, y, data, train_args = prepare("pytorch", mtlalpha=ctc, return_args=True)
    model.eval()
    char_list = train_args.char_list
    lm_args = Namespace(type="lstm", layer=1, unit=2, dropout_rate=0.0)
    lm = LegacyRNNLM(len(char_list), lm_args)
    lm.eval()

    # test previous beam search
    args = Namespace(
        beam_size=3,
        penalty=0.1,            # TODO(karita) non-zero
        ctc_weight=ctc,
        maxlenratio=1.0,
        lm_weight=0.5,
        minlenratio=0,
        nbest=2
    )

    with torch.no_grad():
        feat = x[0, :ilens[0]].numpy()
        nbest = model.recognize(feat, args, char_list, lm.model)
        # print(y[0])
        print(nbest)

    # test new beam search
    decoders = model.decoders
    decoders["lm"] = lm
    decoders["length_bonus"] = LengthBonus()
    weights = dict(decoder=1.0, ctc=args.ctc_weight, lm=args.lm_weight, length_bonus=args.penalty)
    with torch.no_grad():
        enc = model.encode(feat)
        nbest_bs = beam_search(
            x=enc,
            beam_size=args.beam_size,
            weights=weights,
            decoders=decoders,
            token_list=train_args.char_list
        )
        print(nbest_bs)


if __name__ == "__main__":
    test_beam_search_equal()
