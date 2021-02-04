import pytest
import torch

from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.lm.seq_rnn_lm import SequentialRNNLM


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("search_type", ["default", "alsd", "nsc", "tsd"])
@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.parametrize("beam_size", [1, 2])
@pytest.mark.parametrize("nstep", [1, 2])
def test_transducer_decoder_beam_search(
    rnn_type, search_type, use_attention, beam_size, nstep
):
    encoder = torch.randn(10, 4)

    decoder = RNNDecoder(
        10,
        4,
        hidden_size=12,
        embed_pad=0,
        rnn_type=rnn_type,
        num_layers=2,
        use_attention=use_attention,
        use_output_layer=False,
    )
    joint_net = JointNetwork(10, 4, decoder.dunits, joint_space_size=20)

    beam_search = BeamSearchTransducer(
        decoder=decoder,
        joint_network=joint_net,
        beam_size=beam_size,
        lm=None,
        lm_weight=0.0,
        search_type=search_type,
        max_sym_exp=2,
        u_max=10,
        nstep=nstep,
        prefix_alpha=2,
        score_norm=False,
    )

    with torch.no_grad():
        beam_search(encoder)


@pytest.mark.parametrize(
    "decoder_class",
    [
        TransformerDecoder,
        LightweightConvolutionTransformerDecoder,
        LightweightConvolution2DTransformerDecoder,
        DynamicConvolutionTransformerDecoder,
        DynamicConvolution2DTransformerDecoder,
    ],
)
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("search_type", ["default", "alsd", "nsc", "tsd"])
@pytest.mark.parametrize("beam_size", [1, 2])
@pytest.mark.parametrize("nstep", [1, 2])
def test_transformer_transducer_decoder_beam_search(
    decoder_class, normalize_before, search_type, beam_size, nstep
):
    encoder = torch.randn(10, 4)

    decoder = decoder_class(
        10,
        4,
        normalize_before=normalize_before,
        use_output_layer=False,
        linear_units=10,
        use_attention=False,
        embed_pad=0,
    )

    joint_net = JointNetwork(10, 4, decoder.dunits, joint_space_size=20)

    beam_search = BeamSearchTransducer(
        decoder=decoder,
        joint_network=joint_net,
        beam_size=beam_size,
        lm=None,
        lm_weight=0.0,
        search_type=search_type,
        max_sym_exp=2,
        u_max=10,
        nstep=nstep,
        prefix_alpha=2,
        score_norm=False,
    )

    with torch.no_grad():
        beam_search(encoder)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("search_type", ["default", "alsd", "nsc", "tsd"])
def test_transducer_beam_search_with_lm(rnn_type, search_type):
    encoder = torch.randn(10, 4)

    decoder = RNNDecoder(
        10,
        4,
        hidden_size=12,
        embed_pad=0,
        rnn_type=rnn_type,
        num_layers=1,
        use_attention=False,
        use_output_layer=False,
    )
    joint_net = JointNetwork(10, 4, decoder.dunits, joint_space_size=20)

    lm_model = SequentialRNNLM(10, rnn_type=rnn_type)

    beam_search = BeamSearchTransducer(
        decoder=decoder,
        joint_network=joint_net,
        beam_size=2,
        lm=lm_model,
        search_type=search_type,
        max_sym_exp=2,
        u_max=10,
        nstep=2,
        prefix_alpha=2,
        score_norm=True,
    )

    with torch.no_grad():
        beam_search(encoder)
