import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.cls.layers.abs_embedding_fusion import AbsEmbeddingFusion
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class AudioTextAttnFusion(AbsEmbeddingFusion):
    """
    Combine audio embeddings and question embeddings using attention.
    """

    def __init__(self, audio_dim, text_dim, hidden_dim, output_dim):
        """
        Args:
            audio_dim: The dimension of the audio embeddings.
            text_dim: The dimension of the question embeddings.
            hidden_dim: The dimension of the hidden layer used in the attention mechanism.
            output_dim: The dimension of the output embeddings.

        """
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def combine(self, embeddings, lengths):
        """
        Args:
            embeddings: A dict of embeddings to combine.
                Embedding shapes are (batch_size, seq_len, embedding_dim).
            lengths: A dict of lengths corresponding to the embeddings.
        Returns:
            A combined embedding, computed using attention.
        """
        assert "text" in embeddings and "audio" in embeddings
        assert "text" in lengths and "audio" in lengths
        assert embeddings["text"].size(0) == embeddings["audio"].size(
            0
        ), "Text, audio and their lengths must have the same batch size"
        assert lengths["text"].size(0) == lengths["audio"].size(
            0
        ), "Text, audio and their lengths must have the same batch size"
        assert embeddings["text"].size(0) == lengths["text"].size(
            0
        ), "Text, audio and their lengths must have the same batch size"

        audio_emb = embeddings["audio"]
        text_emb = embeddings["text"]
        audio_emb = self.audio_linear(audio_emb)  # B x L_a x D
        text_emb = self.text_linear(text_emb)  # B x L_t x D

        attention_scores = torch.bmm(
            text_emb, audio_emb.transpose(-1, -2)
        )  # B x L_t x L_a
        attention_mask = make_pad_mask(lengths=lengths["audio"]).unsqueeze(
            1
        )  # B x 1 x L_a
        attention_scores.masked_fill_(attention_mask, -float("inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)
        combined_emb = torch.bmm(attention_scores, audio_emb)  # B x L_t x D
        combined_emb = self.output_linear(combined_emb)
        return combined_emb, lengths["text"]


class AudioTextConcat(AbsEmbeddingFusion):
    """
    Concatenates audio and text embeddings along the sequence dimension.
    """

    def __init__(self):
        pass  # No learnable parameters needed for concatenation

    def combine(self, embeddings, lengths):
        """
        Concatenates text and audio embeddings along the sequence dimension.

        Args:
            embeddings: A dict of embeddings to concatenate.
                Embedding shapes are (batch_size, seq_len, embedding_dim).
            lengths: A dict of lengths corresponding to the embeddings.

        Returns:
            A concatenated embedding tensor (batch_size, max_seq_len, embedding_dim)
            and the updated lengths tensor.
        """
        assert "text" in embeddings and "audio" in embeddings
        assert "text" in lengths and "audio" in lengths
        assert embeddings["text"].size(0) == embeddings["audio"].size(
            0
        ), "Text, audio and their lengths must have the same batch size"
        assert lengths["text"].size(0) == lengths["audio"].size(
            0
        ), "Text, audio and their lengths must have the same batch size"
        assert embeddings["text"].size(0) == lengths["text"].size(
            0
        ), "Text, audio and their lengths must have the same batch size"

        text_emb = embeddings["text"]  # (B, L_t, D)
        audio_emb = embeddings["audio"]  # (B, L_a, D)
        text_lens = lengths["text"]  # (B,)
        audio_lens = lengths["audio"]  # (B,)

        batch_size, dim = text_emb.size(0), text_emb.size(-1)
        assert dim == audio_emb.size(
            -1
        ), "Text and audio embeddings must have the same dimension"

        encoder_out_lens = text_lens + audio_lens  # (B,)
        max_len = encoder_out_lens.max()

        encoder_out = torch.zeros(
            (batch_size, max_len, dim), dtype=text_emb.dtype, device=text_emb.device
        )

        for i in range(batch_size):
            text_len = text_lens[i].item()
            audio_len = audio_lens[i].item()

            encoder_out[i, :text_len] = text_emb[i, :text_len]
            encoder_out[i, text_len : text_len + audio_len] = audio_emb[i, :audio_len]

        return encoder_out, encoder_out_lens
