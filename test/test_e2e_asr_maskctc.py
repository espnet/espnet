import argparse
import importlib
import logging
import pytest
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_arg(**kwargs):
    defaults = dict(
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
        transformer_decoder_selfattn_layer_type="selfattn",
        transformer_encoder_selfattn_layer_type="selfattn",
        transformer_init="pytorch",
        transformer_input_layer="conv2d",
        transformer_length_normalized_loss=False,
        report_cer=False,
        report_wer=False,
        mtlalpha=0.3,
        lsm_weight=0.001,
        wshare=4,
        char_list=["<blank>", "a", "e", "i", "o", "u", "<eos>", "<mask>"],
        ctc_type="warpctc",
        decoder_mode="maskctc",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(args):
    idim = 40
    odim = len(args.char_list)

    T = importlib.import_module(
        "espnet.nets.{}_backend.e2e_asr_transformer".format("pytorch")
    )

    model = T.E2E(idim, odim, args)
    batchsize = 5

    x = torch.randn(batchsize, 40, idim)
    ilens = [40, 30, 20, 15, 10]
    n_token = odim - 2  # w/o <eos>/<sos>, <mask>
    y = (torch.rand(batchsize, 10) * n_token % n_token).long()
    olens = [5, 9, 10, 7, 6]
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = model.ignore_id

    data = []
    for i in range(batchsize):
        data.append(
            (
                "utt%d" % i,
                {
                    "input": [{"shape": [ilens[i], idim]}],
                    "output": [{"shape": [olens[i]]}],
                },
            )
        )

    return model, x, torch.tensor(ilens), y, data


def test_mask():
    args = make_arg()
    model, x, ilens, y, data = prepare(args)

    # check <sos>/<eos>, <mask> position
    n_char = len(args.char_list)
    assert model.sos == n_char - 2
    assert model.eos == n_char - 2
    assert model.mask_token == n_char - 1

    # check mask_uniform
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import mask_uniform

    yi, yo = mask_uniform(y, model.mask_token, model.eos, model.ignore_id)
    assert (
        (yi == model.mask_token).detach().numpy()
        == (yo != model.ignore_id).detach().numpy()
    ).all()


@pytest.mark.parametrize(
    "model_dict",
    [
        ({"maskctc_n_iterations": 1, "maskctc_probability_threshold": 0.0}),
        ({"maskctc_n_iterations": 1, "maskctc_probability_threshold": 0.5}),
        ({"maskctc_n_iterations": 2, "maskctc_probability_threshold": 0.5}),
        ({"maskctc_n_iterations": 0, "maskctc_probability_threshold": 0.5}),
    ],
)
def test_transformer_trainable_and_decodable(model_dict):
    args = make_arg(**model_dict)
    model, x, ilens, y, data = prepare(args)

    # decoding params
    recog_args = argparse.Namespace(
        maskctc_n_iterations=args.maskctc_n_iterations,
        maskctc_probability_threshold=args.maskctc_probability_threshold,
    )
    # test training
    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(x, ilens, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # test attention plot
    attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
    from espnet.nets.pytorch_backend.transformer import plot

    plot.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test")

    # test decoding
    with torch.no_grad():
        nbest = model.recognize_maskctc(
            x[0, : ilens[0]].numpy(), recog_args, args.char_list
        )
        print(y[0])
        print(nbest[0]["yseq"][1:-1])


if __name__ == "__main__":
    test_transformer_trainable_and_decodable(
        {"maskctc_n_iterations": 0, "maskctc_probability_threshold": 0.5}
    )
