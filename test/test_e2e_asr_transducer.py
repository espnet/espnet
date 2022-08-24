# coding: utf-8

import argparse
import json
import tempfile

import numpy as np
import pytest
import torch
from packaging.version import parse as V

import espnet.lm.pytorch_backend.extlm as extlm_pytorch
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr_transducer import E2E
from espnet.nets.pytorch_backend.nets_utils import pad_list

is_torch_1_4_plus = V(torch.__version__) >= V("1.4.0")
is_torch_1_5_plus = V(torch.__version__) >= V("1.5.0")


def get_default_train_args(**kwargs):
    train_defaults = dict(
        etype="vggblstmp",
        elayers=1,
        subsample="1_2_2_1_1",
        eunits=4,
        eprojs=4,
        dtype="lstm",
        dlayers=1,
        dunits=4,
        dec_embed_dim=4,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_embed_decoder=0.0,
        joint_dim=2,
        joint_activation_type="tanh",
        transducer_loss_weight=1.0,
        use_ctc_loss=False,
        ctc_loss_weight=0.0,
        ctc_loss_dropout_rate=0.0,
        use_lm_loss=False,
        lm_loss_weight=0.0,
        use_aux_transducer_loss=False,
        aux_transducer_loss_weight=0.0,
        aux_transducer_loss_enc_output_layers=[],
        use_symm_kl_div_loss=False,
        symm_kl_div_loss_weight=0.0,
        char_list=["a", "b", "c", "d"],
        sym_space="<space>",
        sym_blank="<blank>",
        report_cer=False,
        report_wer=False,
        verbose=0,
        outdir=None,
        rnnlm=None,
        model_module="espnet.nets.pytorch_backend.e2e_asr_transducer:E2E",
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def get_default_recog_args(**kwargs):
    recog_defaults = dict(
        batchsize=0,
        beam_size=1,
        nbest=1,
        verbose=0,
        search_type="default",
        nstep=1,
        max_sym_exp=2,
        prefix_alpha=2,
        u_max=5,
        expansion_gamma=2,
        expansion_beta=0.2,
        score_norm_transducer=True,
        rnnlm=None,
        lm_weight=0.1,
    )
    recog_defaults.update(kwargs)

    return argparse.Namespace(**recog_defaults)


def get_default_scope_inputs():
    idim = 15
    odim = 4

    ilens = [12, 8]
    olens = [8, 4]

    return idim, odim, ilens, olens


def get_lm():
    n_layers = 1
    n_units = 4

    char_list = ["<blank>", "<space>", "a", "b", "c", "d", "<eos>"]

    rnnlm = lm_pytorch.ClassifierWithState(
        lm_pytorch.RNNLM(len(char_list), n_layers, n_units, typ="lstm")
    )

    return rnnlm


def get_wordlm():
    n_layers = 1
    n_units = 8

    char_list = ["<blank>", "<space>", "a", "b", "c", "d", "<eos>"]
    word_list = ["<blank>", "<unk>", "ab", "id", "ac", "bd", "<eos>"]

    char_dict = {x: i for i, x in enumerate(char_list)}
    word_dict = {x: i for i, x in enumerate(word_list)}

    word_rnnlm = lm_pytorch.ClassifierWithState(
        lm_pytorch.RNNLM(len(word_list), n_layers, n_units)
    )
    word_rnnlm = lm_pytorch.ClassifierWithState(
        extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor, word_dict, char_dict)
    )

    return word_rnnlm


def prepare_inputs(idim, odim, ilens, olens, is_cuda=False):
    np.random.seed(1)

    feats = [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens]
    labels = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    feats_len = np.array([x.shape[0] for x in feats], dtype=np.int32)

    feats = pad_list([torch.from_numpy(x).float() for x in feats], 0)
    labels = pad_list([torch.from_numpy(y).long() for y in labels], -1)

    feats_len = torch.from_numpy(feats_len).long()

    if is_cuda:
        feats = feats.cuda()
        labels = labels.cuda()
        feats_len = feats_len.cuda()

    return feats, feats_len, labels


@pytest.mark.parametrize(
    "train_dic, recog_dic",
    [
        ({}, {}),
        ({"eprojs": 4}, {}),
        ({"dlayers": 2}, {}),
        ({"etype": "gru"}, {}),
        ({"etype": "blstm"}, {}),
        ({"etype": "blstmp", "elayers": 2, "eprojs": 4}, {}),
        ({"etype": "vgggru"}, {}),
        ({"etype": "vggbru"}, {}),
        ({"etype": "vgggrup", "elayers": 2, "eprojs": 4}, {}),
        ({"dtype": "gru"}, {}),
        ({"dtype": "bgrup"}, {}),
        ({"dtype": "gru", "dlayers": 2}, {}),
        ({"joint-activation-type": "relu"}, {}),
        ({"joint-activation-type": "swish"}, {}),
        ({}, {"score_norm_transducer": False}),
        ({"report_cer": True, "report_wer": True}, {}),
        ({}, {"nbest": 2}),
        ({}, {"beam_size": 1}),
        ({}, {"beam_size": 2}),
        ({}, {"beam_size": 2, "search_type": "nsc"}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 2, "prefix_alpha": 1}),
        ({}, {"beam_size": 2, "search_type": "tsd"}),
        ({}, {"beam_size": 2, "search_type": "tsd", "max-sym-exp": 3}),
        ({}, {"beam_size": 2, "search_type": "alsd"}),
        ({}, {"beam_size": 2, "search_type": "alsd", "u_max": 10}),
        ({}, {"beam_size": 2, "search_type": "maes", "nstep": 2}),
        (
            {},
            {
                "beam_size": 2,
                "search_type": "default",
                "rnnlm": get_wordlm(),
                "lm_weight": 1.0,
            },
        ),
        ({}, {"beam_size": 2, "search_type": "nsc", "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "rnnlm": get_wordlm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 2, "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 2, "rnnlm": get_wordlm()}),
        (
            {},
            {
                "beam_size": 2,
                "search_type": "alsd",
                "rnnlm": get_lm(),
                "lm_weight": 0.2,
            },
        ),
        (
            {},
            {
                "beam_size": 2,
                "search_type": "alsd",
                "rnnlm": get_wordlm(),
                "lm_weight": 0.6,
            },
        ),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_wordlm()}),
        (
            {},
            {"beam_size": 2, "search_type": "maes", "nstep": 2, "rnnlm": get_wordlm()},
        ),
    ],
)
def test_pytorch_transducer_trainable_and_decodable(train_dic, recog_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()

    train_args = get_default_train_args(**train_dic)
    recog_args = get_default_recog_args(**recog_dic)

    model = E2E(idim, odim, train_args)

    batch = prepare_inputs(idim, odim, ilens, olens)

    # to avoid huge training time, cer/wer report
    # is only enabled at validation steps
    if train_args.report_cer or train_args.report_wer:
        model.training = False

    loss = model(*batch)
    loss.backward()

    beam_search = BeamSearchTransducer(
        decoder=model.dec,
        joint_network=model.transducer_tasks.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
    )

    with torch.no_grad():
        in_data = np.random.randn(20, idim)

        model.recognize(in_data, beam_search)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "train_dic",
    [
        {"report_cer": True, "report_wer": True},
    ],
)
@pytest.mark.execution_timeout(3.2)
def test_pytorch_multi_gpu_trainable(train_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(**train_dic)

    ngpu = 2
    device_ids = list(range(ngpu))

    model = E2E(idim, odim, train_args)
    model = torch.nn.DataParallel(model, device_ids)
    model.cuda()

    batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=True)

    loss = 1.0 / ngpu * model(*batch)
    loss.backward(loss.new_ones(ngpu))


def test_calculate_plot_attention():
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args()

    model = E2E(idim, odim, train_args)

    batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=False)

    assert model.calculate_all_attentions(*batch) == []


@pytest.mark.parametrize(
    "train_dic",
    [
        {
            "elayers": 3,
            "use_aux_transducer_loss": True,
            "aux_transducer_loss_enc_output_layers": [1],
        },
        {
            "elayers": 2,
            "use_ctc_loss": True,
            "ctc_loss_weight": 0.5,
            "ctc_loss_dropout_rate": 0.1,
        },
        {
            "etype": "vggblstm",
            "elayers": 3,
            "use_aux_transducer_loss": True,
            "aux_transducer_loss": True,
            "use_symm_kl_div_loss": True,
            "symm_kl_div_loss_weight": 0.5,
            "aux_transducer_loss_enc_output_layers": [0, 1],
        },
        {"dlayers": 2, "use_lm_loss": True, "lm_loss_weight": 0.5},
    ],
)
def test_auxiliary_task(train_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()

    train_args = get_default_train_args(**train_dic)
    recog_args = get_default_recog_args()

    model = E2E(idim, odim, train_args)

    batch = prepare_inputs(idim, odim, ilens, olens)

    loss = model(*batch)
    loss.backward()

    beam_search = BeamSearchTransducer(
        decoder=model.dec,
        joint_network=model.transducer_tasks.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
    )

    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    torch.save(model.state_dict(), tmpdir + "/model.dummy.best")

    with open(tmpdir + "/model.json", "wb") as f:
        f.write(
            json.dumps(
                (idim, odim, vars(train_args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )

    with torch.no_grad():
        in_data = np.random.randn(20, idim)

        model, _ = load_trained_model(tmpdir + "/model.dummy.best", training=False)

        model.recognize(in_data, beam_search)


def test_invalid_aux_transducer_loss_enc_layers():
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(use_aux_transducer_loss=True)

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)

    train_args = get_default_train_args(
        use_aux_transducer_loss=True, aux_transducer_loss_enc_output_layers="foo"
    )

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)

    train_args = get_default_train_args(
        use_aux_transducer_loss=True, aux_transducer_loss_enc_output_layers=[0, 4]
    )

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)

    train_args = get_default_train_args(
        use_aux_transducer_loss=True,
        use_symm_kl_div_loss=True,
        aux_transducer_loss_enc_output_layers=[0],
        elayers=3,
        etype="blstmp",
        subsample="1_2_1",
    )

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)


@pytest.mark.parametrize(
    "train_dic",
    [
        {},
        {"etype": "vggblstm"},
    ],
)
@pytest.mark.parametrize(
    "recog_dic",
    [
        {},
        {"beam_size": 2, "search_type": "default"},
        {"beam_size": 2, "search_type": "alsd"},
        {"beam_size": 2, "search_type": "tsd"},
        {"beam_size": 2, "search_type": "nsc"},
        {"beam_size": 2, "search_type": "maes"},
    ],
)
@pytest.mark.parametrize(
    "quantize_dic",
    [
        {"mod": {torch.nn.Linear}, "dtype": torch.qint8},
        {"mod": {torch.nn.Linear}, "dtype": torch.float16},
        {"mod": {torch.nn.LSTM}, "dtype": torch.qint8},
        {"mod": {torch.nn.LSTM}, "dtype": torch.float16},
        {"mod": {torch.nn.Linear, torch.nn.LSTM}, "dtype": torch.qint8},
        {"mod": {torch.nn.Linear, torch.nn.LSTM}, "dtype": torch.float16},
    ],
)
def test_dynamic_quantization(train_dic, recog_dic, quantize_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()

    train_args = get_default_train_args(**train_dic)
    recog_args = get_default_recog_args(**recog_dic)

    model = E2E(idim, odim, train_args)

    if not is_torch_1_5_plus and (
        torch.nn.Linear in quantize_dic["mod"]
        and quantize_dic["dtype"] == torch.float16
    ):
        # In recognize(...) from asr.py we raise ValueError however
        # AssertionError is originaly raised by torch.
        with pytest.raises(AssertionError):
            model = torch.quantization.quantize_dynamic(
                model,
                quantize_dic["mod"],
                dtype=quantize_dic["dtype"],
            )
        pytest.skip("Skip rest of the test after checking AssertionError")
    else:
        model = torch.quantization.quantize_dynamic(
            model,
            quantize_dic["mod"],
            quantize_dic["dtype"],
        )

    beam_search = BeamSearchTransducer(
        decoder=model.dec,
        joint_network=model.transducer_tasks.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
        quantization=True,
    )

    with torch.no_grad():
        in_data = np.random.randn(20, idim)

        if not is_torch_1_4_plus and torch.nn.LSTM in quantize_dic["mod"]:
            # Cf. previous comment
            with pytest.raises(AssertionError):
                model.recognize(in_data, beam_search)
        else:
            model.recognize(in_data, beam_search)


@pytest.mark.parametrize(
    "train_dic, subsample",
    [
        ({}, 4),
        ({"etype": "blstm"}, 1),
        ({"etype": "blstmp"}, 2),
    ],
)
def test_subsampling(train_dic, subsample):
    idim, odim, ilens, olens = get_default_scope_inputs()

    train_args = get_default_train_args(**train_dic)

    model = E2E(idim, odim, train_args)

    assert model.get_total_subsampling_factor() == subsample
