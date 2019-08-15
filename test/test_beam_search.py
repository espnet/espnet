from argparse import Namespace
import torch

from test.test_e2e_asr_transformer import prepare


def test_beam_search_equal():
    model, x, ilens, y, data = prepare("pytorch")

    # test beam search
    recog_args = Namespace(
        beam_size=1,
        penalty=0.0,
        ctc_weight=0.0,
        maxlenratio=1.0,
        lm_weight=0,
        minlenratio=0,
        nbest=1
    )

    model.eval()
    with torch.no_grad():
        nbest = model.recognize(x[0, :ilens[0]].numpy(), recog_args)
        print(y[0])
        print(nbest[0]["yseq"][1:-1])


if __name__ == "__main__":
    test_beam_search_equal()
