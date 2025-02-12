import pytest
import torch

from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsRandomTokenizer,
    BeatsTokenizer,
    BeatsTokenizerConfig,
    BeatsTokenizerPretrainingPredictor,
    EmbeddingEMA,
    NormEMAVectorQuantizer,
)


@pytest.mark.parametrize("n_codes", [15, 20, 200])
def test_beats_tokenizer_encode(n_codes):
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.encoder_layers = 2
    tokenizer_config.quant_n = n_codes
    tokenizer = BeatsTokenizer(tokenizer_config=vars(tokenizer_config))
    x = torch.randn(2, 16000)
    x_len = torch.LongTensor([16000, 12000])
    token_ids, loss, quantized_features, qlen = tokenizer.encode(xs_pad=x, ilens=x_len)
    assert token_ids.shape[0] == 2
    assert token_ids.min() >= 0
    assert token_ids.max() < n_codes
    assert loss.dim() == 0
    assert quantized_features.shape == (2, 48, tokenizer_config.quant_dim)
    assert qlen.shape == (2,)


def test_embedding_ema_forward():
    emb_ema = EmbeddingEMA(5, 10)  # 5 codes, 10 dim
    emb_idx = torch.randint(0, 5, (2,))
    emb = emb_ema(emb_idx)
    assert emb.shape == (2, 10)


def test_norm_ema_quantizer():
    model = NormEMAVectorQuantizer(
        n_embed=5, embedding_dim=10, beta=1.0, kmeans_init=True
    )
    z = torch.randn((2, 3, 10), requires_grad=True)  # B, n_patch, dim
    z_q, loss, encoding_indices = model(z)
    assert z_q.shape == (2, 3, 10)
    assert encoding_indices.shape == (2, 3)
    assert encoding_indices.min() >= 0 and encoding_indices.max() < 5
    loss.backward()


def test_forward_and_backward_beats_pretraining_predictor():
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.decoder_layers = 3
    predictor = BeatsTokenizerPretrainingPredictor(
        tokenizer_config=vars(tokenizer_config),
    )
    # B,T_patch,quant_dim
    x = torch.randn((2, 500, tokenizer_config.quant_dim), requires_grad=True)
    x_lens = torch.LongTensor([450, 500])
    y = predictor(x, x_lens)
    y.sum().backward()
    assert y.shape == (2, 500, tokenizer_config.decoder_embed_dim)
    assert x.grad is not None


@pytest.mark.parametrize("n_codes", [5, 1024])
def test_beats_random_tokenizer_encode(n_codes):
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.quant_n = n_codes
    tokenizer = BeatsRandomTokenizer(tokenizer_config=vars(tokenizer_config))
    x = torch.randn(2, 160_000)
    x_len = torch.LongTensor([160_000, 120_000])
    token_ids, _, _, token_id_len = tokenizer.encode(xs_pad=x, ilens=x_len)
    assert token_ids.shape[0] == 2
    assert token_ids.min() >= 0
    assert token_ids.max() < n_codes
    assert tuple(token_id_len) == (496, 368)


def test_beats_random_tokenizer_var_length():
    # Padding of other elements should not affect each other
    arr = torch.randn(3, 176_000)
    arrlen = torch.LongTensor([96_000, 80_000, 160_000])
    model = BeatsRandomTokenizer()
    token_ids1, _, _, token_id_len1 = model.encode(arr, arrlen)

    for i in range(3):
        arr_ = arr[i : i + 1]  # drop everything else
        arrlen_ = arrlen[i : i + 1]
        token_ids2, _, _, token_id_len2 = model.encode(arr_, arrlen_)

        assert token_id_len1[i] == token_id_len2[0]
        # Match token ids
        assert torch.all(
            token_ids1[i, : token_id_len1[i]] == token_ids2[0, : token_id_len2[0]]
        )
