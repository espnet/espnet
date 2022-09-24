# coding: utf-8

# Copyright 2018 Hiroshi Seki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import re

import numpy
import pytest
import torch


def make_arg(**kwargs):
    defaults = dict(
        aconv_chans=10,
        aconv_filts=100,
        adim=320,
        aheads=4,
        apply_uttmvn=False,
        atype="location",
        awin=5,
        badim=320,
        batch_bins=0,
        batch_count="auto",
        batch_frames_in=0,
        batch_frames_inout=0,
        batch_frames_out=0,
        batch_size=10,
        bdropout_rate=0.0,
        beam_size=3,
        blayers=2,
        bnmask=3,
        bprojs=300,
        btype="blstmp",
        bunits=300,
        char_list=["a", "i", "u", "e", "o"],
        context_residual=False,
        ctc_type="builtin",
        ctc_weight=0.2,
        dlayers=1,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        dtype="lstm",
        dunits=300,
        elayers_sd=1,
        elayers=2,
        etype="vggblstmp",
        eprojs=100,
        eunits=100,
        fbank_fmax=None,
        fbank_fmin=0.0,
        fbank_fs=16000,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        nbest=5,
        maxlenratio=1.0,
        minlenratio=0.0,
        n_mels=80,
        num_spkrs=1,
        outdir=None,
        penalty=0.5,
        ref_channel=0,
        replace_sos=False,
        report_cer=False,
        report_wer=False,
        sortagrad=0,
        spa=False,
        stats_file=None,
        subsample="1_2_2_1_1",
        sym_blank="<blank>",
        sym_space="<space>",
        tgt_lang=False,
        use_beamformer=False,
        use_dnn_mask_for_wpe=False,
        use_frontend=False,
        use_wpe=False,
        uttmvn_norm_means=False,
        uttmvn_norm_vars=False,
        verbose=2,
        wdropout_rate=0.0,
        weight_decay=0.0,
        wlayers=2,
        wpe_delay=3,
        wpe_taps=5,
        wprojs=300,
        wtype="blstmp",
        wunits=300,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        p.data[:] = val


@pytest.mark.parametrize(
    ("etype", "dtype", "num_spkrs", "spa", "m_str", "text_idx1"),
    [
        ("vggblstmp", "lstm", 2, True, "espnet.nets.pytorch_backend.e2e_asr_mix", 0),
        ("vggbgrup", "gru", 2, True, "espnet.nets.pytorch_backend.e2e_asr_mix", 1),
    ],
)
def test_recognition_results_multi_outputs(
    etype, dtype, num_spkrs, spa, m_str, text_idx1
):
    const = 1e-4
    numpy.random.seed(1)

    # ctc_weight: 0.5 (hybrid CTC/attention), cannot be 0.0 (attention) or 1.0 (CTC)
    for text_idx2, ctc_weight in enumerate([0.5]):
        args = make_arg(
            etype=etype, ctc_weight=ctc_weight, num_spkrs=num_spkrs, spa=spa
        )
        m = importlib.import_module(m_str)
        model = m.E2E(40, 5, args)

        if "pytorch" in m_str:
            init_torch_weight_const(model, const)
        else:
            init_chainer_weight_const(model, const)

        data = [
            (
                "aaa",
                dict(
                    feat=numpy.random.randn(100, 40).astype(numpy.float32),
                    token=["", ""],
                ),
            )
        ]

        in_data = data[0][1]["feat"]
        nbest_hyps = model.recognize(in_data, args, args.char_list)

        for i in range(num_spkrs):
            y_hat = nbest_hyps[i][0]["yseq"][1:]
            seq_hat = [args.char_list[int(idx)] for idx in y_hat]
            seq_hat_text = "".join(seq_hat).replace("<space>", " ")

            assert re.match(r"[aiueo]+", seq_hat_text)


@pytest.mark.parametrize(
    ("etype", "dtype", "num_spkrs", "m_str", "data_idx"),
    [("vggblstmp", "lstm", 2, "espnet.nets.pytorch_backend.e2e_asr_mix", 0)],
)
def test_pit_process(etype, dtype, num_spkrs, m_str, data_idx):
    bs = 10
    m = importlib.import_module(m_str)

    losses_2 = torch.ones([bs, 4], dtype=torch.float32)
    for i in range(bs):
        losses_2[i][i % 4] = 0
    true_losses_2 = torch.ones(bs, dtype=torch.float32) / 2
    perm_choices_2 = [[0, 1], [1, 0], [1, 0], [0, 1]]
    true_perm_2 = []
    for i in range(bs):
        true_perm_2.append(perm_choices_2[i % 4])
    true_perm_2 = torch.tensor(true_perm_2).long()

    losses = [losses_2]
    true_losses = [torch.mean(true_losses_2)]
    true_perm = [true_perm_2]

    args = make_arg(etype=etype, num_spkrs=num_spkrs)
    model = m.E2E(40, 5, args)
    min_loss, min_perm = model.pit.pit_process(losses[data_idx])

    assert min_loss == true_losses[data_idx]
    assert torch.equal(min_perm, true_perm[data_idx])


@pytest.mark.parametrize(
    ("use_frontend", "use_beamformer", "bnmask", "num_spkrs", "m_str"),
    [(True, True, 3, 2, "espnet.nets.pytorch_backend.e2e_asr_mix")],
)
def test_dnn_beamformer(use_frontend, use_beamformer, bnmask, num_spkrs, m_str):
    bs = 4
    m = importlib.import_module(m_str)
    const = 1e-4
    numpy.random.seed(1)

    args = make_arg(
        use_frontend=use_frontend,
        use_beamformer=use_beamformer,
        bnmask=bnmask,
        num_spkrs=num_spkrs,
    )
    model = m.E2E(257, 5, args)
    beamformer = model.frontend.beamformer
    mask_estimator = beamformer.mask

    if "pytorch" in m_str:
        init_torch_weight_const(model, const)
    else:
        init_chainer_weight_const(model, const)

    # STFT feature
    feat_real = torch.from_numpy(numpy.random.uniform(size=(bs, 100, 2, 257))).float()
    feat_imag = torch.from_numpy(numpy.random.uniform(size=(bs, 100, 2, 257))).float()
    feat = m.to_torch_tensor({"real": feat_real, "imag": feat_imag})
    ilens = torch.tensor([100] * bs).long()

    # dnn_beamformer
    enhanced, ilens, mask_speeches = beamformer(feat, ilens)
    assert (bnmask - 1) == len(mask_speeches)
    assert (bnmask - 1) == len(enhanced)

    # beamforming by hand
    feat = feat.permute(0, 3, 2, 1)
    masks, _ = mask_estimator(feat, ilens)
    mask_speech1, mask_speech2, mask_noise = masks

    b = importlib.import_module("espnet.nets.pytorch_backend.frontends.beamformer")

    psd_speech1 = b.get_power_spectral_density_matrix(feat, mask_speech1)
    psd_speech2 = b.get_power_spectral_density_matrix(feat, mask_speech2)
    psd_noise = b.get_power_spectral_density_matrix(feat, mask_noise)

    u1 = torch.zeros(*(feat.size()[:-3] + (feat.size(-2),)), device=feat.device)
    u1[..., args.ref_channel].fill_(1)
    u2 = torch.zeros(*(feat.size()[:-3] + (feat.size(-2),)), device=feat.device)
    u2[..., args.ref_channel].fill_(1)

    ws1 = b.get_mvdr_vector(psd_speech1, psd_speech2 + psd_noise, u1)
    ws2 = b.get_mvdr_vector(psd_speech2, psd_speech1 + psd_noise, u2)

    enhanced1 = b.apply_beamforming_vector(ws1, feat).transpose(-1, -2)
    enhanced2 = b.apply_beamforming_vector(ws2, feat).transpose(-1, -2)

    assert torch.equal(enhanced1.real, enhanced[0].real)
    assert torch.equal(enhanced2.real, enhanced[1].real)
    assert torch.equal(enhanced1.imag, enhanced[0].imag)
    assert torch.equal(enhanced2.imag, enhanced[1].imag)
