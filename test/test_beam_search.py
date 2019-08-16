from argparse import Namespace
import numpy
import pytest
import torch

from espnet.nets.pytorch_backend.beam_search import beam_search
from espnet.nets.pytorch_backend.beam_search import LengthBonus
from espnet.nets.pytorch_backend.e2e_asr import E2E as RNN
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as Transformer
from espnet.nets.pytorch_backend.lm.legacy import LegacyRNNLM


rnn_args = Namespace(
    elayers=1,
    subsample="1_2_2_1_1",
    etype="vggblstm",
    eunits=16,
    eprojs=8,
    dtype="lstm",
    dlayers=1,
    dunits=16,
    atype="location",
    aheads=2,
    awin=5,
    aconv_chans=4,
    aconv_filts=10,
    lsm_type="",
    lsm_weight=0.0,
    sampling_probability=0.0,
    adim=16,
    dropout_rate=0.0,
    dropout_rate_decoder=0.0,
    nbest=5,
    beam_size=2,
    penalty=0.5,
    maxlenratio=1.0,
    minlenratio=0.0,
    ctc_weight=0.2,
    lm_weight=0.0,
    rnnlm=None,
    streaming_min_blank_dur=10,
    streaming_onset_margin=2,
    streaming_offset_margin=2,
    verbose=2,
    outdir=None,
    ctc_type="warpctc",
    report_cer=False,
    report_wer=False,
    sym_space="<space>",
    sym_blank="<blank>",
    sortagrad=0,
    grad_noise=False,
    context_residual=False,
    use_frontend=False,
    replace_sos=False,
    tgt_lang=False
)

transformer_args = Namespace(
    adim=16,
    aheads=2,
    dropout_rate=0.0,
    transformer_attn_dropout_rate=None,
    elayers=2,
    eunits=16,
    dlayers=2,
    dunits=16,
    sym_space="<space>",
    sym_blank="<blank>",
    transformer_init="pytorch",
    transformer_input_layer="conv2d",
    transformer_length_normalized_loss=True,
    report_cer=False,
    report_wer=False,
    ctc_type="warpctc",
    lsm_weight=0.001,
)


# from test.test_e2e_asr_transformer import prepare
def prepare(E2E, args, mtlalpha=0.0):
    args.mtlalpha = mtlalpha
    args.char_list = ['a', 'e', 'i', 'o', 'u']
    idim = 40
    odim = 5
    model = E2E(idim, odim, args)
    batchsize = 5
    x = torch.randn(batchsize, 40, idim)
    ilens = [40, 30, 20, 15, 10]
    n_token = odim - 1
    # avoid 0 for eps in ctc
    y = (torch.rand(batchsize, 10) * n_token % (n_token - 1)).long() + 1
    olens = [3, 9, 10, 2, 3]
    for i in range(batchsize):
        x[i, ilens[i]:] = -1
        y[i, olens[i]:] = -1

    data = []
    for i in range(batchsize):
        data.append(("utt%d" % i, {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}]
        }))
    return model, x, torch.tensor(ilens), y, data, args


@pytest.mark.parametrize("model_class, args", [(Transformer, transformer_args), (RNN, rnn_args)])
def test_beam_search_equal(model_class, args):
    ctc = 0.0                   # TODO(karita) non-zero
    model, x, ilens, y, data, train_args = prepare(model_class, args)
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
        lm_weight=0.5,          # TODO(karita) non-zero
        minlenratio=0,
        nbest=2
    )

    with torch.no_grad():
        feat = x[0, :ilens[0]].numpy()
        nbest = model.recognize(feat, args, char_list, lm.model)
        # print(y[0])
        print(nbest)
        print("===================")

    # test new beam search
    decoders = model.decoders
    decoders["lm"] = lm
    decoders["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(decoder=1.0 - ctc, ctc=ctc, lm=args.lm_weight, length_bonus=args.penalty)
    with torch.no_grad():
        enc = model.encode(feat)
        nbest_bs = beam_search(
            x=enc,
            sos=model.sos,
            eos=model.eos,
            beam_size=args.beam_size,
            weights=weights,
            decoders=decoders,
            token_list=train_args.char_list,
            maxlenratio=args.maxlenratio,
            minlenratio=args.minlenratio,
        )
        print(nbest_bs)
    for i, (expected, actual) in enumerate(zip(nbest, nbest_bs)):
        assert expected["yseq"] == actual["yseq"]
        numpy.testing.assert_allclose(expected["score"], actual["score"], rtol=1e-6)


if __name__ == "__main__":
    test_beam_search_equal(RNN, rnn_args)
    test_beam_search_equal(Transformer, transformer_args)
