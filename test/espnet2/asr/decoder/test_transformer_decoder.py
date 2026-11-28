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
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet2.legacy.nets.beam_search import BeamSearch
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.legacy.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer


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
def test_TransformerDecoder_partially_AR(decoder_class):
    """This test is for partially auto-regressive decoding.

    This function tests if the `expand_kv` works properly for TransformerDecoder.
    """
    decoder = decoder_class(
        vocab_size=5,
        encoder_output_size=8,
        input_layer="embed",
        normalize_before=True,
        use_output_layer=True,
        linear_units=10,
    )

    enc = torch.randn(5, 10, 8)
    tgt = torch.ones(5, 3).type(torch.long)
    tgt_lengths = torch.ones(5).type(torch.long) * 3
    tgt_mask = (~make_pad_mask(tgt_lengths)[:, None, :]).to(enc.device)
    m = subsequent_mask(tgt_mask.size(-1), device=enc.device).unsqueeze(0)
    tgt_mask = tgt_mask & m
    with torch.no_grad():
        decoder.forward_partially_AR(
            tgt,
            tgt_mask,
            tgt_lengths,
            enc,
        )


def test_TransformerDecoder_use_sdpa_covers_source_attention():
    torch.manual_seed(0)
    kwargs = dict(attention_heads=4, linear_units=64, num_blocks=2)
    matmul = TransformerDecoder(50, 32, use_sdpa=False, **kwargs).eval()
    sdpa = TransformerDecoder(50, 32, use_sdpa=True, **kwargs).eval()
    sdpa.load_state_dict(matmul.state_dict())
    assert all(layer.self_attn.use_sdpa for layer in sdpa.decoders)
    assert all(layer.src_attn.use_sdpa for layer in sdpa.decoders)

    hs = torch.randn(3, 40, 32)
    hlens = torch.tensor([40, 31, 7])
    ys = torch.randint(0, 50, (3, 9))
    ylens = torch.tensor([9, 5, 2])
    with torch.no_grad():
        expected, _ = matmul(hs, hlens, ys, ylens)
        actual, _ = sdpa(hs, hlens, ys, ylens)
    for i, length in enumerate(ylens):
        torch.testing.assert_close(actual[i, :length], expected[i, :length])


@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.uint8, torch.bool])
def test_TransformerDecoder_use_sdpa_reads_a_numeric_memory_mask_as_binary(
    mask_dtype,
):
    # The matmul path treats any mask as 1 = attend, 0 = masked (mask.eq(0)).
    # SDPA would add a float mask to the logits instead, so the SDPA path must
    # turn the mask into a boolean first.
    from espnet2.legacy.nets.pytorch_backend.transformer.mask import subsequent_mask

    torch.manual_seed(0)
    kwargs = dict(attention_heads=4, linear_units=64, num_blocks=2)
    matmul = TransformerDecoder(50, 32, use_sdpa=False, **kwargs).eval()
    sdpa = TransformerDecoder(50, 32, use_sdpa=True, **kwargs).eval()
    sdpa.load_state_dict(matmul.state_dict())

    memory = torch.randn(2, 6, 32)
    memory_mask = torch.tensor(
        [[[1, 1, 1, 1, 0, 0]], [[1, 1, 0, 0, 0, 0]]], dtype=mask_dtype
    )
    tgt = torch.randint(0, 50, (2, 4))
    tgt_mask = subsequent_mask(4).unsqueeze(0)
    with torch.no_grad():
        expected, _ = matmul.forward_one_step(
            tgt, tgt_mask, memory, memory_mask=memory_mask
        )
        actual, _ = sdpa.forward_one_step(
            tgt, tgt_mask, memory, memory_mask=memory_mask
        )
    torch.testing.assert_close(actual, expected)
