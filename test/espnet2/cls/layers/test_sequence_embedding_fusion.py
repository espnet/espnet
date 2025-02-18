import pytest
import torch
import torch.nn as nn

from espnet2.cls.layers.sequence_embedding_fusion import (
    AudioTextAttnFusion,
    AudioTextConcat,
)


def get_audio_text_attn_fusion_layer():
    return AudioTextAttnFusion(
        audio_dim=128, text_dim=256, hidden_dim=64, output_dim=32
    )


def test_audio_text_attn_combiner_initialization():
    combine_layer = get_audio_text_attn_fusion_layer()
    assert isinstance(combine_layer.audio_linear, nn.Linear)
    assert isinstance(combine_layer.text_linear, nn.Linear)
    assert isinstance(combine_layer.output_linear, nn.Linear)
    assert combine_layer.audio_linear.in_features == 128
    assert combine_layer.text_linear.in_features == 256
    assert combine_layer.audio_linear.out_features == 64
    assert combine_layer.text_linear.out_features == 64
    assert combine_layer.output_linear.in_features == 64
    assert combine_layer.output_linear.out_features == 32


def test_audio_text_attn_combine_shapes():
    combine_layer = get_audio_text_attn_fusion_layer()
    batch_size = 4
    seq_len_audio = 10
    seq_len_text = 5
    audio_dim = 128
    text_dim = 256

    embeddings = {
        "audio": torch.randn(batch_size, seq_len_audio, audio_dim),
        "text": torch.randn(batch_size, seq_len_text, text_dim),
    }
    lengths = {
        "audio": torch.LongTensor([10, 8, 6, 5]),
        "text": torch.LongTensor([5, 4, 3, 2]),
    }

    combined_emb, output_lengths = combine_layer(embeddings, lengths)
    assert combined_emb.shape == (batch_size, seq_len_text, 32)
    assert torch.all(output_lengths == lengths["text"])


def test_audio_text_attn_combiner_attention_masking():
    combine_layer = get_audio_text_attn_fusion_layer()
    batch_size = 2
    seq_len_audio = 6
    seq_len_text = 4
    audio_dim = 128
    text_dim = 256

    embeddings = {
        "audio": torch.randn(batch_size, seq_len_audio, audio_dim),
        "text": torch.randn(batch_size, seq_len_text, text_dim),
    }
    lengths = {"audio": torch.LongTensor([6, 3]), "text": torch.LongTensor([4, 2])}

    combined_emb, _ = combine_layer(embeddings, lengths)
    assert not torch.isnan(combined_emb).any()


def test_audio_text_attn_combiner_attention_scores_validity():
    combine_layer = get_audio_text_attn_fusion_layer()
    batch_size = 3
    seq_len_audio = 7
    seq_len_text = 5
    audio_dim = 128
    text_dim = 256

    embeddings = {
        "audio": torch.randn(batch_size, seq_len_audio, audio_dim),
        "text": torch.randn(batch_size, seq_len_text, text_dim),
    }
    lengths = {
        "audio": torch.LongTensor([7, 5, 4]),
        "text": torch.LongTensor([5, 3, 2]),
    }

    combined_emb, _ = combine_layer(embeddings, lengths)

    assert combined_emb.shape == (batch_size, seq_len_text, 32)
    assert not torch.isnan(combined_emb).any()
    assert torch.isfinite(combined_emb).all()


def test_audio_text_concat_basic():
    fusion = AudioTextConcat()

    text_emb = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )  # (2, 2, 2)
    audio_emb = torch.tensor([[[9.0, 10.0]], [[11.0, 12.0]]])  # (2, 1, 2)

    embeddings = {"text": text_emb, "audio": audio_emb}
    lengths = {"text": torch.LongTensor([2, 2]), "audio": torch.LongTensor([1, 1])}

    combined_emb, combined_lens = fusion(embeddings, lengths)

    expected_emb = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]], [[5.0, 6.0], [7.0, 8.0], [11.0, 12.0]]]
    )
    expected_lens = torch.LongTensor([3, 3])

    assert torch.allclose(combined_emb, expected_emb)
    assert torch.equal(combined_lens, expected_lens)


def test_audio_text_concat_varying_lengths():
    fusion = AudioTextConcat()

    text_emb = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    audio_emb = torch.tensor([[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]])  # (1, 3, 2)

    embeddings = {"text": text_emb, "audio": audio_emb}
    lengths = {"text": torch.LongTensor([2]), "audio": torch.LongTensor([3])}

    combined_emb, combined_lens = fusion(embeddings, lengths)

    expected_emb = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]
    )
    expected_lens = torch.LongTensor([5])

    assert torch.allclose(combined_emb, expected_emb)
    assert torch.equal(combined_lens, expected_lens)


def test_audio_text_concat_different_embedding_dims():
    fusion = AudioTextConcat()

    text_emb = torch.randn((1, 2, 4))  # (1, 2, 4)
    audio_emb = torch.randn((1, 2, 3))  # (1, 2, 3) --> Dimension mismatch

    embeddings = {"text": text_emb, "audio": audio_emb}
    lengths = {"text": torch.LongTensor([2]), "audio": torch.LongTensor([2])}

    with pytest.raises(
        AssertionError, match="Text and audio embeddings must have the same dimension"
    ):
        fusion(embeddings, lengths)


def test_audio_text_concat_batch_size_mismatch():
    fusion = AudioTextConcat()

    text_emb = torch.randn((2, 3, 4))  # (2, 3, 4)
    audio_emb = torch.randn((1, 2, 4))  # (1, 2, 4) --> Batch size mismatch

    embeddings = {"text": text_emb, "audio": audio_emb}
    lengths = {
        "text": torch.LongTensor([3, 3]),
        "audio": torch.LongTensor([2]),
    }

    with pytest.raises(
        AssertionError,
        match="Text, audio and their lengths must have the same batch size",
    ):
        fusion(embeddings, lengths)
