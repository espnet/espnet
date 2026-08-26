import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.wavlm_encoder import TorchAudioWavLMPretrainEncoder

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


def _build(**kwargs):
    return TorchAudioWavLMPretrainEncoder(
        20,
        extractor_conv_layer_config=[[3, 3, 2]],
        encoder_pos_conv_kernel=16,
        encoder_pos_conv_groups=4,
        encoder_embed_dim=4,
        encoder_num_layers=1,
        encoder_num_heads=1,
        encoder_ff_interm_features=4,
        encoder_num_buckets=8,
        encoder_max_distance=16,
        num_classes=10,
        final_dim=10,
        **kwargs,
    )


@pytest.mark.parametrize(
    "finetuning, eval, freeze_encoder_updates",
    [
        (False, False, 0),
        (True, False, 0),
        (True, False, 1),
        (True, True, 0),
    ],
)
def test_Encoder_forward_backward(finetuning, eval, freeze_encoder_updates):
    if not is_torch_1_12_1_plus:
        return

    encoder = _build(
        finetuning=finetuning, freeze_encoder_updates=freeze_encoder_updates
    )
    x = torch.randn(2, 32, requires_grad=True)
    y = torch.randint(low=0, high=10, size=(2, 15), dtype=torch.long)
    x_lens = torch.LongTensor([32, 16])
    y, _, p = encoder(x, x_lens, y)
    if not eval:
        encoder.train()
        if not finetuning:
            p.sum().backward()
        else:
            if freeze_encoder_updates == 0:
                y.sum().backward()
            else:
                y.sum()  # requires_grad=False if freezing
    else:
        encoder.eval()
        y, _, p = encoder(x, x_lens, y)
        y.sum()


def test_Encoder_output_size():
    if not is_torch_1_12_1_plus:
        return
    assert _build().output_size() == 4


def test_Encoder_reload_params():
    if not is_torch_1_12_1_plus:
        return
    encoder = _build()
    encoder.reload_pretrained_parameters()


def test_Encoder_has_relative_position_bias():
    """WavLM's defining architectural change over HuBERT."""
    if not is_torch_1_12_1_plus:
        return
    encoder = _build()
    assert any("rel_attn_embed" in k for k in encoder.state_dict())


def test_Transformer_masks_padded_keys():
    """WavLM self-attention must not attend to padded frames.

    torchaudio's WavLMSelfAttention rejects the additive attention mask its
    Wav2Vec2/HuBERT counterpart takes and wants a key padding mask instead,
    which nothing upstream of it supplies. WavLMEncoder/WavLMTransformer reroute
    the mask; without that, padded frames would leak into every valid frame.
    """
    if not is_torch_1_12_1_plus:
        return
    encoder = _build().eval()
    transformer = encoder.wavlm_pretrain_model.wav2vec2.encoder.transformer

    feats = torch.randn(1, 20, 4)
    valid = 12
    padded = feats.clone()
    padded[:, valid:] = 0.0
    key_padding_mask = torch.arange(20).expand(1, 20) >= torch.LongTensor([valid])[
        :, None
    ]

    short = transformer(feats[:, :valid], attention_mask=None)
    masked = transformer(padded, attention_mask=key_padding_mask)
    torch.testing.assert_close(masked[:, :valid], short, atol=1e-5, rtol=1e-4)

    # Sanity check that the mask is what makes the difference.
    unmasked = transformer(padded, attention_mask=None)
    assert not torch.allclose(unmasked[:, :valid], short, atol=1e-5)


def test_Encoder_padding_is_masked():
    """A padded batch encodes its valid frames the same as an unpadded one.

    Uses ``extractor_mode="layer_norm"``: the default ``group_norm`` normalizes
    over the time axis in the first convolution block, so the feature extractor
    is not length-invariant under zero padding. That is inherent to the
    wav2vec2/HuBERT extractor and identical for
    ``TorchAudioHuBERTPretrainEncoder``; it is not what this test is about.
    """
    if not is_torch_1_12_1_plus:
        return
    encoder = _build(extractor_mode="layer_norm", finetuning=True).eval()
    x = torch.randn(1, 64)
    x_lens = torch.LongTensor([32])

    short, len_short, _ = encoder(x[:, :32], x_lens)
    padded, len_padded, _ = encoder(x, x_lens)
    assert torch.equal(len_short, len_padded)
    torch.testing.assert_close(short, padded[:, : short.size(1)], atol=1e-5, rtol=1e-4)
