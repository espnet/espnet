import pytest
import torch

from espnet2.beats.tokenizer import (
    BeatsRandomTokenizer,
    BeatsTokenizer,
    BeatsTokenizerConfig,
    BeatsTokenizerPretrainingPredictor,
    EmbeddingEMA,
    NormEMAVectorQuantizer,
)


@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize("n_codes", [15, 20, 200])
def test_beats_tokenizer_encode(n_codes):
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.encoder_layers = 2
    tokenizer_config.quant_n = n_codes
    tokenizer = BeatsTokenizer(tokenizer_config=vars(tokenizer_config))
    x = torch.randn(2, 16000)
    x_len = torch.LongTensor([16000, 12000])
    result = tokenizer.encode(xs_pad=x, ilens=x_len)
    token_ids = result["codes"]
    loss = result["embed_loss"]
    quantized_features = result["quantize_feature"]
    qlen = result["code_lengths"]
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


@pytest.mark.execution_timeout(60)
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


@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize("n_codes", [5, 1024])
def test_beats_random_tokenizer_encode(n_codes):
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.quant_n = n_codes
    tokenizer = BeatsRandomTokenizer(tokenizer_config=vars(tokenizer_config))
    # Short clips (~1s and ~0.75s) — algorithm verification only, no acoustic
    # meaning; cheaper to run on CI than 10s inputs.
    x = torch.randn(2, 16_000)
    x_len = torch.LongTensor([16_000, 12_000])
    result = tokenizer.encode(xs_pad=x, ilens=x_len)
    token_ids = result["codes"]
    token_id_len = result["code_lengths"]
    assert token_ids.shape[0] == 2
    assert token_ids.min() >= 0
    assert token_ids.max() < n_codes
    assert token_id_len.shape == (2,)
    assert token_id_len[0] > token_id_len[1] > 0


@pytest.mark.execution_timeout(60)
def test_beats_random_tokenizer_var_length():
    # Padding of other elements should not affect each other.
    # Short clips (~0.4 / 0.8 / 1 s); only invariance under padding is checked.
    arr = torch.randn(3, 17_600)
    arrlen = torch.LongTensor([int(16_000 * 0.4), int(16_000 * 0.8), 16_000])
    model = BeatsRandomTokenizer()
    result1 = model.encode(arr, arrlen)
    token_ids1 = result1["codes"]
    token_id_len1 = result1["code_lengths"]

    for i in range(3):
        arr_ = arr[i : i + 1]  # drop everything else
        arrlen_ = arrlen[i : i + 1]
        result2 = model.encode(arr_, arrlen_)
        token_ids2 = result2["codes"]
        token_id_len2 = result2["code_lengths"]

        assert token_id_len1[i] == token_id_len2[0]
        # Match token ids
        assert torch.all(
            token_ids1[i, : token_id_len1[i]] == token_ids2[0, : token_id_len2[0]]
        )


def test_norm_ema_quantizer_all_reduces_after_late_ddp_init(monkeypatch):
    """Codebook EMA uses global stats when DDP starts after model build."""
    generator = torch.Generator().manual_seed(1)
    codebook = torch.nn.functional.normalize(
        torch.randn(4, 3, generator=generator), dim=-1
    )
    inputs = [torch.randn(1, 50, 3, generator=generator) for _ in range(2)]

    def build_quantizer():
        quantizer = NormEMAVectorQuantizer(
            n_embed=4, embedding_dim=3, beta=1.0, decay=0.5, kmeans_init=True
        )
        quantizer.embedding.weight.copy_(codebook)
        quantizer.embedding.initted.fill_(True)
        return quantizer.train()

    # A single process seeing both ranks' data computes the global EMA update.
    expected = build_quantizer()
    expected(torch.cat(inputs, dim=1))

    # Built before the process group exists, as Lightning does.
    quantizer = build_quantizer()

    # Statistics the other rank contributes for inputs[1]: bins, then embed_sum.
    other = torch.nn.functional.normalize(inputs[1], dim=-1).reshape(-1, 3)
    other_codes = torch.cdist(other, codebook).argmin(dim=1)
    other_encodings = torch.nn.functional.one_hot(other_codes, 4).to(other.dtype)
    contributions = [other_encodings.sum(0), other.t() @ other_encodings]

    def fake_all_reduce(tensor):
        tensor.add_(contributions.pop(0))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    quantizer(inputs[0])

    assert contributions == []
    torch.testing.assert_close(quantizer.embedding.weight, expected.embedding.weight)
