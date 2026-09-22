from test.espnet2.universa.test_metric_tokenizer import token_info

import numpy as np
import pytest
import torch

from espnet2.universa.ar_universa import ARUniversa
from espnet2.universa.ar_universa.data import ARMetricCollateFn, ARMetricProcessor


@pytest.fixture
def make_arecho():
    def build(**kwargs):
        info = token_info()
        options = dict(
            input_size=8,
            metric2id={"mos": 0, "language": 1},
            metric2type={"mos": "numerical", "language": "categorical"},
            metric_vocab_size=len(info["VOCAB"]) + 4,
            metric_token_info=info,
            embedding_size=8,
            embedding_dim=256,
            use_rope=True,
            use_ref_audio=False,
            use_ref_text=False,
            use_normalize=False,
            audio_encoder_params=dict(
                num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
            ),
            metric_decoder_params=dict(
                num_blocks=1, attention_heads=2, linear_units=16
            ),
            text_encoder_params=dict(
                num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
            ),
            cross_attention_params=dict(n_head=2, dropout_rate=0.0),
        )
        options.update(kwargs)
        return ARUniversa(**options)

    return build


@pytest.mark.parametrize("pool_size", [1, 10])
def test_arecho_backward_and_constrained_search(make_arecho, pool_size):
    info = token_info()
    model = make_arecho()
    processor = ARMetricProcessor(
        metric_token_info=info, metrics_list=["mos", "language"], train=True
    )
    samples = [
        (
            "a",
            processor(
                "a",
                dict(
                    audio=np.random.randn(10, 8).astype(np.float32),
                    metrics={"mos": 1.5, "language": "eng"},
                ),
            ),
        )
    ]
    # Test the tokenizer and collator separately from waveform preprocessing.
    samples[0][1]["audio"] = np.random.randn(10, 8).astype(np.float32)
    _, batch = ARMetricCollateFn()(samples)
    loss, stats, _ = model(**batch)
    assert torch.isfinite(loss) and loss > 0
    torch.testing.assert_close(loss, stats["loss"])
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert model.sequential_metrics
    assert model.sos == 2 and model.eos == 3  # Published ARECHO convention.
    model.eval()
    model.set_inference(
        pool_size, ["language", "mos"], True, True, use_fixed_order=True
    )
    result = model.inference(batch["audio"], batch["audio_lengths"])
    assert result["language"][0] in ["eng", "jpn"]
    assert result["mos"][0] in info["tokenizer"]["mos"]

    assert result["token_seq"][0][1::2] == [10, 4]


@pytest.mark.parametrize("reference", ["ref_audio", "ref_text"])
def test_reference_requires_lengths(make_arecho, reference):
    model = make_arecho(use_ref_audio=True, use_ref_text=True, vocab_size=6)
    ref = torch.randn(1, 5, 8) if reference == "ref_audio" else torch.ones(1, 5).long()
    with pytest.raises(ValueError, match=reference + "_lengths"):
        model.encode(torch.randn(1, 7, 8), torch.tensor([7]), **{reference: ref})


@pytest.mark.parametrize(
    "audio_ref,text_ref", [(False, False), (True, False), (False, True), (True, True)]
)
def test_reference_encoding_preserves_inputs(make_arecho, audio_ref, text_ref):
    model = make_arecho(
        use_ref_audio=True, use_ref_text=True, vocab_size=6, use_normalize=True
    ).eval()
    audio = torch.randn(2, 7, 8)
    lengths = torch.tensor([7, 5])
    refs = {}
    if audio_ref:
        refs.update(
            ref_audio=torch.randn(2, 6, 8), ref_audio_lengths=torch.tensor([6, 4])
        )
    if text_ref:
        refs.update(
            ref_text=torch.tensor([[1, 2, 3], [4, 5, -1]]),
            ref_text_lengths=torch.tensor([3, 2]),
        )
    original = {key: value.clone() for key, value in dict(audio=audio, **refs).items()}
    encoded, encoded_lengths = model.encode(audio, lengths, **refs)
    assert encoded.shape == (2, 7, 24)
    torch.testing.assert_close(encoded_lengths, lengths)
    for key, value in dict(audio=audio, **refs).items():
        torch.testing.assert_close(value, original[key])
    if not audio_ref:
        assert torch.count_nonzero(encoded[:, :, 8:16]) == 0
    if not text_ref:
        assert torch.count_nonzero(encoded[:, :, 16:]) == 0
    again, _ = model.encode(audio, lengths, **refs)
    torch.testing.assert_close(encoded, again)
    encoded.square().mean().backward()
    assert model.audio_encoder.embed[0].weight.grad is not None


def test_empty_metric_targets(make_arecho):
    model = make_arecho()
    _, batch = ARMetricCollateFn()(
        [("a", dict(audio=np.random.randn(5, 8).astype(np.float32), metrics={}))]
    )
    loss, stats, _ = model(**batch)
    assert torch.isfinite(loss) and loss > 0
    assert stats["value_ar_decoder"] == 0
    loss.backward()


def test_target_padding_uses_lengths(make_arecho):
    model = make_arecho().eval()
    audio = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 4])
    targets = torch.tensor([[4, 7, 10, 12], [4, 7, 0, 0]])
    metrics = dict(metric_token=targets, metric_token_lengths=torch.tensor([4, 2]))
    loss, _, _ = model(audio, lengths, metrics)
    # Values outside the declared length must never become decoder targets.
    dirty = targets.clone()
    dirty[1, 2:] = 12
    other, _, _ = model(audio, lengths, dict(metrics, metric_token=dirty))
    torch.testing.assert_close(loss, other)
    assert dirty[1, 2:].tolist() == [12, 12]


@pytest.mark.parametrize("randomize", [False, True])
def test_collate_sparse_metrics_preserves_pairs(randomize, monkeypatch):
    monkeypatch.setattr("random.shuffle", lambda pairs: pairs.reverse())
    labels = {"mos": (4, 7), "language": (10, 12)}
    samples = [
        ("a", dict(audio=np.ones((5, 8), dtype=np.float32), metrics=labels)),
        ("b", dict(audio=np.ones((3, 8), dtype=np.float32))),
    ]
    keys, batch = ARMetricCollateFn(randomize=randomize)(samples)
    assert keys == ["a", "b"]
    assert batch["audio_lengths"].tolist() == [5, 3]
    assert batch["metrics"]["metric_token_lengths"].tolist() == [4, 0]
    expected = [10, 12, 4, 7] if randomize else [4, 7, 10, 12]
    assert batch["metrics"]["metric_token"].tolist() == [expected, [0, 0, 0, 0]]
    assert list(labels.values()) == [(4, 7), (10, 12)]
    _, unlabelled = ARMetricCollateFn()([samples[1]])
    assert "metrics" not in unlabelled


@pytest.mark.parametrize("skip_label_score", [False, True])
@pytest.mark.parametrize("fixed_order", [False, True])
def test_inference_metric_subset(make_arecho, skip_label_score, fixed_order):
    model = make_arecho().eval()
    model.set_inference(2, ["mos"], skip_label_score, True, fixed_order)
    result = model.inference(torch.randn(1, 5, 8), torch.tensor([5]))
    assert "mos" in result and "language" not in result
    assert result["token_seq"][0][1::2] == [4]
    with pytest.raises(ValueError, match="Invalid token"):
        model.set_inference(1, ["unknown"], True)
