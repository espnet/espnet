import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import (  # noqa: H301
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.ctc import CTCPrefixScorer


@pytest.mark.parametrize("input_layer", ["linear", "embed"])
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("use_output_layer", [True, False])
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
def test_TransformerDecoder_backward(
    input_layer, normalize_before, use_output_layer, decoder_class
):
    decoder = decoder_class(
        10,
        12,
        input_layer=input_layer,
        normalize_before=normalize_before,
        use_output_layer=use_output_layer,
        linear_units=10,
    )
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    if input_layer == "embed":
        t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    else:
        t = torch.randn(2, 4, 10)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


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
def test_TransformerDecoder_init_state(decoder_class):
    decoder = decoder_class(10, 12)
    x = torch.randn(9, 12)
    state = decoder.init_state(x)
    t = torch.randint(0, 10, [4], dtype=torch.long)
    decoder.score(t, state, x)


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
def test_TransformerDecoder_invalid_type(decoder_class):
    with pytest.raises(ValueError):
        decoder_class(10, 12, input_layer="foo")


@pytest.mark.parametrize("input_layer", ["embed"])
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("use_output_layer", [True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("maxlenratio", [1.0, 0.0, -1.0])
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
def test_TransformerDecoder_beam_search(
    input_layer, normalize_before, use_output_layer, dtype, maxlenratio, decoder_class
):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)
    encoder_output_size = 4

    decoder = decoder_class(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        input_layer=input_layer,
        normalize_before=normalize_before,
        use_output_layer=use_output_layer,
        linear_units=10,
    )
    beam = BeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": decoder},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, encoder_output_size).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=maxlenratio,
            minlenratio=0.0,
        )


@pytest.mark.parametrize("input_layer", ["embed"])
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("use_output_layer", [True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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
def test_TransformerDecoder_batch_beam_search(
    input_layer, normalize_before, use_output_layer, dtype, decoder_class
):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)
    encoder_output_size = 4

    decoder = decoder_class(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        input_layer=input_layer,
        normalize_before=normalize_before,
        use_output_layer=use_output_layer,
        linear_units=10,
    )
    beam = BatchBeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": decoder},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, encoder_output_size).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=0.0,
            minlenratio=0.0,
        )


@pytest.mark.parametrize("input_layer", ["embed"])
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("use_output_layer", [True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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
def test_TransformerDecoder_batch_beam_search_online(
    input_layer, normalize_before, use_output_layer, dtype, decoder_class, tmp_path
):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)
    encoder_output_size = 8

    decoder = decoder_class(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        input_layer=input_layer,
        normalize_before=normalize_before,
        use_output_layer=use_output_layer,
        linear_units=10,
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size)
    ctc.to(dtype)
    ctc_scorer = CTCPrefixScorer(ctc=ctc, eos=vocab_size - 1)
    beam = BatchBeamSearchOnlineSim(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 0.7, "ctc": 0.3},
        scorers={"test": decoder, "ctc": ctc_scorer},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    cp = tmp_path / "config.yaml"
    yp = tmp_path / "dummy.yaml"
    with cp.open("w") as f:
        f.write("config: " + str(yp) + "\n")
    with yp.open("w") as f:
        f.write("encoder_conf:\n")
        f.write("    block_size: 4\n")
        f.write("    hop_size: 2\n")
        f.write("    look_ahead: 1\n")
    beam.set_streaming_config(cp)
    with cp.open("w") as f:
        f.write("encoder_conf:\n")
        f.write("    block_size: 4\n")
        f.write("    hop_size: 2\n")
        f.write("    look_ahead: 1\n")
    beam.set_streaming_config(cp)
    beam.set_block_size(4)
    beam.set_hop_size(2)
    beam.set_look_ahead(1)
    beam.to(dtype=dtype)

    enc = torch.randn(10, encoder_output_size).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=0.0,
            minlenratio=0.0,
        )
