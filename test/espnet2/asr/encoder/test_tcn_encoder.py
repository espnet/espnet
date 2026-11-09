import pytest
import torch

from espnet2.asr.encoder.abs_look_ahead_encoder import AbsLookAheadEncoder
from espnet2.asr.encoder.tcn_encoder import TCNEncoder
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm
from espnet2.torch_utils.check_look_ahead import check_look_ahead

D_IN = 8
SMALL = dict(
    output_size=6, bottleneck_size=4, hidden_size=8, num_blocks=3, num_repeats=2
)
T = 60


def _encoder(**kwargs):
    """A small encoder in eval mode whose norms have random affine parameters.

    With the initial norm parameters (bias 0; BN statistics mean 0, variance 1)
    a padded frame of zeros stays zero through a norm, which would hide a read
    of the padding; random ones make it nonzero.
    """
    torch.manual_seed(0)
    encoder = TCNEncoder(D_IN, **{**SMALL, **kwargs}).eval()
    with torch.no_grad():
        for m in encoder.modules():
            if isinstance(m, (ChannelwiseLayerNorm, torch.nn.BatchNorm1d)):
                for p in m.parameters():
                    p.copy_(1.0 + 0.5 * torch.randn_like(p))
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_()
                m.running_var.uniform_(0.5, 1.5)
    return encoder


def _run(encoder, x, lens=None):
    if lens is None:
        lens = torch.full((x.size(0),), x.size(1))
    with torch.no_grad():
        return encoder(x, lens)[0]


@pytest.mark.parametrize("norm_type", ["cLN", "BN"])
@pytest.mark.parametrize("future_taps, look_ahead", [(0, 0), (2, 0), (2, 14)])
def test_forward_backward(norm_type, future_taps, look_ahead):
    encoder = TCNEncoder(
        D_IN,
        norm_type=norm_type,
        future_taps=future_taps,
        look_ahead=look_ahead,
        **SMALL,
    )
    x = torch.randn(2, 20, D_IN, requires_grad=True)
    lens = torch.tensor([20, 13])
    y, olens, states = encoder(x, lens)
    assert y.shape == (2, 20, 6)
    assert torch.equal(olens, lens)
    assert states is None
    assert torch.all(y[1, 13:] == 0)
    y.sum().backward()


def test_is_a_look_ahead_encoder():
    encoder = _encoder()
    assert isinstance(encoder, AbsLookAheadEncoder)
    assert encoder.output_size() == 6


@pytest.mark.parametrize("causal", [None, True])
@pytest.mark.parametrize("norm_type", ["gLN", "GN"])
def test_rejects_time_pooling_norm_with_a_bounded_look_ahead(norm_type, causal):
    with pytest.raises(ValueError, match="causal=False"):
        TCNEncoder(D_IN, norm_type=norm_type, causal=causal, **SMALL)
    with pytest.raises(ValueError):
        TCNEncoder(D_IN, norm_type=norm_type, future_taps=2, **SMALL)


@pytest.mark.parametrize("norm_type", ["gLN", "GN"])
def test_legacy_noncausal_time_pooling_norm(norm_type):
    """causal=False with gLN/GN builds, as in the previous encoder.

    Every output reads the whole sequence: the look-ahead stays at the full
    kernel, and receptive_field and forward_streaming raise.
    """
    encoder = TCNEncoder(D_IN, norm_type=norm_type, causal=False, **SMALL).eval()
    assert encoder.look_ahead == encoder.max_look_ahead
    encoder.set_look_ahead(encoder.max_look_ahead)
    with pytest.raises(ValueError):
        encoder.set_look_ahead(0)
    with pytest.raises(RuntimeError):
        encoder.receptive_field
    with pytest.raises(RuntimeError):
        encoder.forward_streaming(torch.randn(1, 4, D_IN))
    y, _, _ = encoder(torch.randn(2, 20, D_IN), torch.tensor([20, 20]))
    assert y.shape == (2, 20, 6)


def test_contexts():
    # dilations 1, 2, 4 repeated twice; kernel 3 gives 2 past taps per block
    encoder = _encoder(kernel_size=3, future_taps=2)
    assert encoder.past_context == 2 * 2 * (1 + 2 + 4)
    assert encoder.max_look_ahead == 2 * 2 * (1 + 2 + 4)
    assert _encoder(future_taps=0).max_look_ahead == 0
    # the current frame and the past context
    assert encoder.receptive_field == 1 + 2 * 2 * (1 + 2 + 4)


@pytest.mark.parametrize("future_taps", [0, 2])
def test_online_reads_exactly_the_receptive_field(future_taps):
    """At look-ahead 0 output t reads input frames t - receptive_field + 1 to t.

    Rerunning the encoder on the last receptive_field frames gives the output
    of the last frame of the whole sequence (what a streaming caller does).
    """
    encoder = _encoder(future_taps=future_taps)
    rf = encoder.receptive_field
    torch.manual_seed(5)
    x = torch.randn(1, T, D_IN)
    y = _run(encoder, x)
    t = T - 1
    x2 = x.clone()
    x2[:, : t - rf + 1] = torch.randn(1, t - rf + 1, D_IN) * 10
    assert torch.equal(_run(encoder, x2)[:, t], y[:, t])
    x2[:, t - rf + 1] = torch.randn(1, D_IN) * 10  # cLN removes a constant shift
    assert not torch.equal(_run(encoder, x2)[:, t], y[:, t])
    tail = _run(encoder, x[:, t - rf + 1 :])
    torch.testing.assert_close(tail[:, -1], y[:, t], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("future_taps", [0, 2])
@pytest.mark.parametrize("norm_type", ["cLN", "BN"])
def test_online_never_reads_ahead(future_taps, norm_type):
    """Look-ahead 0: no output reads a frame after it.

    For every t0, changing the frames from t0 on leaves every output before t0
    bit-identical, and output t0 changes.
    """
    encoder = _encoder(future_taps=future_taps, norm_type=norm_type)
    assert encoder.look_ahead == 0
    torch.manual_seed(1)
    check_look_ahead(
        lambda x: _run(encoder, x), torch.randn(1, T, D_IN), lambda i: i, range(1, T)
    )


def test_online_does_not_use_the_future_taps():
    """Look-ahead 0 gives the same output whatever the future tap weights are."""
    encoder = _encoder(future_taps=2)
    x = torch.randn(1, T, D_IN)
    y = _run(encoder, x)
    with torch.no_grad():
        for block in encoder.blocks:
            block.depthwise.weight[..., block.kernel_size :] = 100.0
    assert torch.equal(_run(encoder, x), y)


@pytest.mark.parametrize("look_ahead", [1, 3, 8, 14, 28])
def test_offline_reads_exactly_the_look_ahead(look_ahead):
    """Output t depends on input frame t + look_ahead and on no later frame."""
    encoder = _encoder(future_taps=2, look_ahead=look_ahead)
    assert encoder.look_ahead == look_ahead
    torch.manual_seed(2)
    check_look_ahead(
        lambda x: _run(encoder, x),
        torch.randn(1, T, D_IN),
        lambda i: i + look_ahead,
        range(look_ahead + 1, T),
    )


def test_every_look_ahead_up_to_the_maximum_is_reachable():
    encoder = _encoder(future_taps=2)
    for frames in range(encoder.max_look_ahead + 1):
        encoder.set_look_ahead(frames)
        taps = encoder.future_taps_per_block()
        dilations = [b.dilation for b in encoder.blocks]
        assert sum(t * d for t, d in zip(taps, dilations)) == frames
        assert all(0 <= t <= 2 for t in taps)


@pytest.mark.parametrize("frames", [-1, 29])
def test_look_ahead_out_of_range_raises(frames):
    with pytest.raises(ValueError):
        _encoder(future_taps=2).set_look_ahead(frames)


def test_unreachable_look_ahead_raises():
    # one block at dilation 4 with one future tap: only 0 and 4 are reachable
    encoder = TCNEncoder(D_IN, num_blocks=1, num_repeats=1, future_taps=1)
    encoder.blocks[0].dilation = 4
    encoder.set_look_ahead(4)
    with pytest.raises(ValueError):
        encoder.set_look_ahead(2)


def test_modes_share_the_weights():
    """Changing the look-ahead changes neither the parameters nor the state dict."""
    encoder = _encoder(future_taps=2)
    params = {k: v.clone() for k, v in encoder.state_dict().items()}
    ids = [id(p) for p in encoder.parameters()]
    for frames in (0, 14, 28, 0):
        encoder.set_look_ahead(frames)
        assert [id(p) for p in encoder.parameters()] == ids
        state = encoder.state_dict()
        assert state.keys() == params.keys()
        for k in params:
            assert torch.equal(state[k], params[k])


def test_checkpoint_loads_in_both_modes():
    """A state dict saved in one mode gives the same outputs in every mode."""
    trained = _encoder(future_taps=2, look_ahead=28)
    fresh = TCNEncoder(D_IN, future_taps=2, look_ahead=0, **SMALL).eval()
    fresh.load_state_dict(trained.state_dict())
    x = torch.randn(1, T, D_IN)
    for frames in (0, 14, 28):
        trained.set_look_ahead(frames)
        fresh.set_look_ahead(frames)
        assert torch.equal(_run(trained, x), _run(fresh, x))


@pytest.mark.parametrize("look_ahead", [0, 14, 28])
def test_batched_equals_alone(look_ahead):
    """A short utterance padded beside a longer one encodes as it does alone."""
    encoder = _encoder(future_taps=2, look_ahead=look_ahead)
    torch.manual_seed(3)
    x = torch.randn(2, T, D_IN)
    x[1, 37:] = torch.randn(T - 37, D_IN)  # padding with any values
    lens = torch.tensor([T, 37])
    batched = _run(encoder, x, lens)
    alone = _run(encoder, x[1:, :37], torch.tensor([37]))
    torch.testing.assert_close(batched[1, :37], alone[0], rtol=1e-5, atol=1e-6)
    assert torch.all(batched[1, 37:] == 0)


@pytest.mark.parametrize("norm_type", ["cLN", "BN"])
@pytest.mark.parametrize("future_taps, look_ahead", [(0, 0), (2, 0), (2, 5), (2, 28)])
def test_padded_batch_equals_each_utterance_alone(norm_type, future_taps, look_ahead):
    """Every utterance of a padded batch encodes as it does alone.

    The padding is zeroed before each depthwise convolution, so the future taps
    of a shorter utterance read zeros after its end, as its own zero padding
    gives alone, and not what the earlier layers computed on the padding.
    """
    encoder = _encoder(
        future_taps=future_taps, look_ahead=look_ahead, norm_type=norm_type
    )
    torch.manual_seed(4)
    lengths = [40, 23, 31]
    x = torch.randn(3, 40, D_IN)
    out = _run(encoder, x, torch.tensor(lengths))
    for b, n in enumerate(lengths):
        alone = _run(encoder, x[b : b + 1, :n], torch.tensor([n]))
        torch.testing.assert_close(out[b, :n], alone[0], rtol=1e-4, atol=1e-5)
        assert torch.all(out[b, n:] == 0)


def _conv_tasnet(causal, norm_type="cLN", pre_output_nonlinear="prelu"):
    """The previous TCNEncoder's network, random weights and norm statistics."""
    from espnet2.enh.layers.tcn import TemporalConvNet

    torch.manual_seed(7)
    net = TemporalConvNet(
        N=D_IN,
        B=4,
        H=8,
        P=3,
        X=3,
        R=2,
        C=1,
        out_channel=6,
        norm_type=norm_type,
        causal=causal,
        pre_mask_nonlinear=pre_output_nonlinear,
        mask_nonlinear="linear",
    ).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_()
                m.running_var.uniform_(0.5, 1.5)
    return net


@pytest.mark.parametrize("pre_output_nonlinear", ["prelu", "linear", "tanh"])
@pytest.mark.parametrize(
    "causal, norm_type",
    [(True, "cLN"), (True, "BN"), (False, "cLN"), (False, "BN")]
    + [(False, "gLN"), (False, "GN")],
)
def test_previous_encoder_config_and_checkpoint_load(
    causal, norm_type, pre_output_nonlinear
):
    """A checkpoint of the previous TCNEncoder loads and gives the same output.

    That encoder took causal (kernel_size counted all the taps) and kept
    espnet2.enh.layers.tcn.TemporalConvNet as ``tcn``; its state dict, under a
    parent module's prefix as in a model checkpoint, loads strictly.
    """
    net = _conv_tasnet(causal, norm_type, pre_output_nonlinear)
    old = {f"encoder.tcn.{k}": v for k, v in net.state_dict().items()}
    model = torch.nn.Module()
    model.encoder = TCNEncoder(
        D_IN,
        causal=causal,
        norm_type=norm_type,
        pre_output_nonlinear=pre_output_nonlinear,
        **SMALL,
    ).eval()
    model.load_state_dict(old)  # strict: every key renamed, none missing
    encoder = model.encoder
    assert encoder.max_look_ahead == (0 if causal else 2 * (1 + 2 + 4))
    assert encoder.look_ahead == encoder.max_look_ahead
    x = torch.randn(2, T, D_IN)
    with torch.no_grad():
        expected = net(x.transpose(1, 2)).squeeze(1).transpose(1, 2)
    torch.testing.assert_close(_run(encoder, x), expected, rtol=1e-5, atol=1e-5)


def test_causal_excludes_future_taps():
    with pytest.raises(ValueError):
        TCNEncoder(D_IN, causal=True, future_taps=2)
    with pytest.raises(ValueError):
        TCNEncoder(D_IN, causal=False, kernel_size=4)


@pytest.mark.parametrize("norm_type", ["cLN", "BN"])
@pytest.mark.parametrize("future_taps, look_ahead", [(0, 0), (2, 0), (2, 28)])
@pytest.mark.parametrize(
    "splits",
    [[1] * T, [1, 2, 3, 5, 8, 13, 28], [17, 1, 1, 40, 1], [T]],
    ids=["k1", "fibonacci", "irregular", "whole"],
)
def test_streaming_equals_online_forward(norm_type, future_taps, look_ahead, splits):
    """Chunks of any sizes, fed with the state, give forward() at look-ahead 0.

    The look-ahead set on the encoder does not matter: streaming is online.
    """
    assert sum(splits) == T
    encoder = _encoder(future_taps=future_taps, norm_type=norm_type)
    torch.manual_seed(6)
    x = torch.randn(2, T, D_IN)
    expected = _run(encoder, x)  # look-ahead 0
    encoder.set_look_ahead(look_ahead)
    state, outs, t = None, [], 0
    with torch.no_grad():
        for k in splits:
            out, state = encoder.forward_streaming(x[:, t : t + k], state)
            assert out.shape == (2, k, 6)
            # bounded: receptive_field - 1 frames over all blocks
            assert sum(c.size(2) for c in state) == encoder.receptive_field - 1
            outs.append(out)
            t += k
    torch.testing.assert_close(torch.cat(outs, dim=1), expected, rtol=1e-5, atol=1e-6)
    assert encoder.look_ahead == look_ahead


def test_streaming_with_kernel_size_one_has_no_state():
    encoder = _encoder(kernel_size=1)
    x = torch.randn(1, 10, D_IN)
    with torch.no_grad():
        out, state = encoder.forward_streaming(x[:, :3])
        out2, state = encoder.forward_streaming(x[:, 3:], state)
    assert all(c.size(2) == 0 for c in state)
    torch.testing.assert_close(torch.cat([out, out2], dim=1), _run(encoder, x))


def test_look_ahead_0_equals_the_causal_encoder():
    """Look-ahead 0 is the encoder without future taps, with the same weights."""
    dual = _encoder(future_taps=2, look_ahead=0)
    causal = TCNEncoder(D_IN, future_taps=0, **SMALL).eval()
    state = dual.state_dict()
    for i, block in enumerate(dual.blocks):
        key = f"blocks.{i}.depthwise.weight"
        state[key] = state[key][..., : block.kernel_size]  # drop the future taps
    causal.load_state_dict(state)
    x = torch.randn(2, T, D_IN)
    lens = torch.tensor([T, 31])
    torch.testing.assert_close(_run(dual, x, lens), _run(causal, x, lens))


def test_look_ahead_set_later_equals_look_ahead_at_construction():
    built = _encoder(future_taps=2, look_ahead=14)
    switched = _encoder(future_taps=2, look_ahead=0)
    switched.set_look_ahead(28)
    switched.set_look_ahead(14)
    assert switched.future_taps_per_block() == built.future_taps_per_block()
    x = torch.randn(1, T, D_IN)
    assert torch.equal(_run(switched, x), _run(built, x))


@pytest.mark.parametrize("look_ahead", [0, 4, 28])
def test_unused_future_taps_get_no_gradient(look_ahead):
    """Only the future taps the look-ahead uses are trained by that mode."""
    encoder = _encoder(future_taps=2, look_ahead=look_ahead).train()
    x = torch.randn(2, T, D_IN)
    encoder(x, torch.tensor([T, 31]))[0].pow(2).sum().backward()
    for block in encoder.blocks:
        grad = block.depthwise.weight.grad
        used = block.kernel_size + block.future_taps
        assert torch.all(grad[..., used:] == 0)
        assert torch.any(grad[..., :used] != 0)


def test_positional_arguments_keep_the_previous_order():
    """norm_type, causal and pre_output_nonlinear follow num_repeats, as before."""
    encoder = TCNEncoder(D_IN, 6, 4, 8, 3, 3, 2, "BN", True, "linear")
    assert isinstance(encoder.blocks[0].norm_in, torch.nn.BatchNorm1d)
    assert encoder.max_look_ahead == 0
    assert isinstance(encoder.output_act, torch.nn.Identity)
    with pytest.raises(TypeError):
        TCNEncoder(D_IN, 6, 4, 8, 3, 3, 2, "cLN", None, "prelu", 2)


def test_streaming_with_batch_norm_needs_eval_mode():
    encoder = _encoder(norm_type="BN").train()
    with pytest.raises(RuntimeError):
        encoder.forward_streaming(torch.randn(1, 4, D_IN))
    _encoder(norm_type="cLN").train().forward_streaming(torch.randn(1, 4, D_IN))


@pytest.mark.parametrize("ignore_init_mismatch", [False, True])
def test_previous_checkpoint_initializes_an_encoder_with_future_taps(
    tmp_path, ignore_init_mismatch
):
    """--init_param with a causal checkpoint of the previous encoder.

    The keys are converted before load_pretrained_model compares them, and the
    future taps start at zero: every look-ahead gives the causal network's
    output until they are trained.
    """
    from espnet2.torch_utils.load_pretrained_model import load_pretrained_model

    net = _conv_tasnet(causal=True)
    path = tmp_path / "previous.pth"
    torch.save({f"encoder.tcn.{k}": v for k, v in net.state_dict().items()}, path)
    model = torch.nn.Module()
    model.encoder = TCNEncoder(D_IN, future_taps=2, **SMALL).eval()
    load_pretrained_model(str(path), model, ignore_init_mismatch)
    x = torch.randn(1, T, D_IN)
    with torch.no_grad():
        expected = net(x.transpose(1, 2)).squeeze(1).transpose(1, 2)
    for frames in (0, 14, 28):
        model.encoder.set_look_ahead(frames)
        torch.testing.assert_close(
            _run(model.encoder, x), expected, rtol=1e-5, atol=1e-5
        )
