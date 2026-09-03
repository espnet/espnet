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


def _count_memory_projections(decoder):
    """Count the key projections of the first layer's source attention."""
    calls = []
    handle = decoder.decoders[0].src_attn.linear_k.register_forward_hook(
        lambda *_: calls.append(1)
    )
    return calls, handle


def test_TransformerDecoder_memory_kv_is_projected_once_per_memory():
    """The encoder output is projected on the first call and then reused."""
    torch.manual_seed(0)
    decoder = TransformerDecoder(10, 12, num_blocks=2, linear_units=10).eval()
    calls, handle = _count_memory_projections(decoder)
    memory = torch.randn(3, 9, 12)
    ys = torch.randint(0, 10, (3, 2))
    with torch.no_grad():
        _, states = decoder.batch_score(ys, [None] * 3, memory)
        assert len(calls) == 1
        decoder.batch_score(torch.cat([ys, ys[:, :1]], 1), states, memory)
        decoder.batch_score(torch.cat([ys, ys[:, :1]], 1), states, memory)
        assert len(calls) == 1, "the same memory was projected again"
    handle.remove()


def test_TransformerDecoder_memory_kv_cache_is_invalidated():
    torch.manual_seed(0)
    decoder = TransformerDecoder(10, 12, num_blocks=1, linear_units=10).eval()
    calls, handle = _count_memory_projections(decoder)
    ys = torch.randint(0, 10, (2, 2))
    memory = torch.randn(2, 9, 12)
    with torch.no_grad():
        decoder.batch_score(ys, [None] * 2, memory)
        # in-place modification bumps the version counter
        memory.mul_(0.5)
        decoder.batch_score(ys, [None] * 2, memory)
        assert len(calls) == 2
        # another tensor of the same shape is another memory, even if the
        # allocator hands out the same address
        other = memory.clone()
        decoder.batch_score(ys, [None] * 2, other)
        assert len(calls) == 3
        # the cache dies with the tensor it was computed from
        del other
        import gc

        gc.collect()
        assert getattr(decoder, "_memory_kv_cache", None) is None
    handle.remove()


@pytest.mark.parametrize("with_mask", [False, True])
def test_TransformerDecoder_batch_score_shared_memory(with_mask):
    """One encoder row per utterance scores like one row per hypothesis.

    This is the layout `BatchBeamSearch.score_full` hands to a scorer with
    `accepts_shared_memory`: hypothesis `b * n_hyp + i` belongs to utterance
    `b`, so the memory need not be replicated over the beam.
    """
    torch.manual_seed(0)
    n_utt, n_hyp, xlen = 2, 3, 9
    decoder = TransformerDecoder(10, 12, num_blocks=2, linear_units=10).eval()
    memory = torch.randn(n_utt, xlen, 12)
    replicated = memory.repeat_interleave(n_hyp, dim=0)
    mask = None
    if with_mask:
        mask = make_pad_mask(torch.tensor([xlen, xlen - 4]), maxlen=xlen)
        mask = (~mask).unsqueeze(1).repeat_interleave(n_hyp, dim=0)
    ys = torch.randint(0, 10, (n_utt * n_hyp, 3))
    with torch.no_grad():
        ref, ref_states = decoder.batch_score(
            ys, [None] * len(ys), replicated, xs_mask=mask
        )
        out, states = decoder.batch_score(ys, [None] * len(ys), memory, xs_mask=mask)
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
        # and again with a cache, from the second step on
        ys2 = torch.cat([ys, ys[:, :1]], dim=1)
        ref2, _ = decoder.batch_score(ys2, ref_states, replicated, xs_mask=mask)
        out2, _ = decoder.batch_score(ys2, states, memory, xs_mask=mask)
        torch.testing.assert_close(out2, ref2, rtol=1e-6, atol=1e-6)


def test_TransformerDecoder_expanded_memory_is_projected_once():
    """A (T, D) encoder output expanded over the beam is one projection."""
    torch.manual_seed(0)
    decoder = TransformerDecoder(10, 12, num_blocks=1, linear_units=10).eval()
    x = torch.randn(9, 12)
    linear_k = decoder.decoders[0].src_attn.linear_k
    seen = []
    handle = linear_k.register_forward_hook(lambda m, inp, out: seen.append(inp[0]))
    ys = torch.randint(0, 10, (4, 2))
    with torch.no_grad():
        ref, _ = decoder.batch_score(ys, [None] * 4, x.unsqueeze(0).repeat(4, 1, 1))
        assert seen[-1].size(0) == 4
        out, _ = decoder.batch_score(ys, [None] * 4, x.expand(4, 9, 12))
        assert seen[-1].size(0) == 1, "a stride-0 memory should be projected once"
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
        # the hypothesis set shrinks as the beam search goes; the projections
        # of the same encoder output are still reused
        n = len(seen)
        decoder.batch_score(ys[:2], [None] * 2, x.expand(2, 9, 12))
        decoder.batch_score(ys[:1], [None] * 1, x.expand(1, 9, 12))
        assert len(seen) == n
        # the single-hypothesis `score` path of `BeamSearch` reuses them too
        n = len(seen)
        decoder.score(ys[0], None, x)
        decoder.score(ys[1], None, x)
        assert len(seen) == n
    handle.remove()
