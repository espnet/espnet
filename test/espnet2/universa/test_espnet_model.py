import numpy as np
import pytest
import torch


@pytest.mark.parametrize("multi_branch", [False, True])
@pytest.mark.parametrize("references", [False, True])
def test_frontend_wrapper_training_stats_and_checkpoint(
    make_model, multi_branch, references
):
    from espnet2.asr.frontend.default import DefaultFrontend
    from espnet2.universa.espnet_model import ESPnetUniversaModel

    predictor = make_model(
        multi_branch=multi_branch,
        use_ref_audio=references,
        use_ref_text=references,
        vocab_size=5,
        text_encoder_params=dict(
            num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
        ),
    )
    model = ESPnetUniversaModel(
        predictor, DefaultFrontend(n_fft=64, hop_length=16, n_mels=8)
    )
    audio, lengths = torch.randn(2, 256), torch.tensor([256, 200])
    extra = (
        dict(
            ref_audio=torch.randn(2, 256),
            ref_audio_lengths=lengths,
            ref_text=torch.tensor([[1, 2, 3], [2, -1, -1]]),
            ref_text_lengths=torch.tensor([3, 1]),
        )
        if references
        else {}
    )
    original_text = extra.get("ref_text", torch.empty(0)).clone()
    loss, stats, weight = model(
        audio, lengths, {"mos": torch.tensor([1.0, 2.0])}, **extra
    )
    expected_stats = (
        {
            f"{metric}_{loss}"
            for metric in ("mos", "wer")
            for loss in ("mse", "l1", "overall")
        }
        if multi_branch
        else {"mse", "l1"}
    )
    assert set(stats) == expected_stats | {"loss"}
    assert torch.isfinite(loss) and weight == 2
    loss.backward()
    assert predictor.audio_encoder.embed[0].weight.grad is not None
    features = model.collect_feats(audio, lengths, **extra)
    assert features["audio"] is audio
    assert ("ref_audio" in features) == references
    torch.testing.assert_close(extra.get("ref_text", torch.empty(0)), original_text)
    model.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        result = model.inference(audio, lengths, **extra)
        absent = model.inference(audio, lengths)
    assert np.isfinite(result["mos"]).all() and np.isfinite(absent["mos"]).all()
    assert result["encoded_feat"].shape[-1] == (24 if references else 8)


def test_reference_free_predictor_uses_abstract_defaults():
    from espnet2.universa.abs_universa import AbsUniversa
    from espnet2.universa.espnet_model import ESPnetUniversaModel

    class ReferenceFreePredictor(AbsUniversa):
        def forward(self, audio, audio_lengths, metrics, **kwargs):
            assert not kwargs
            return audio.sum(), {}, audio.new_tensor(len(audio))

        def inference(self, audio, audio_lengths, **kwargs):
            assert not kwargs
            return {"encoded_feat": audio}

    model = ESPnetUniversaModel(ReferenceFreePredictor(), None)
    assert not model.use_ref_audio and not model.use_ref_text
    audio, lengths = torch.randn(2, 10, 8), torch.tensor([10, 8])
    loss, _, weight = model(audio, lengths, {})
    torch.testing.assert_close(loss, audio.sum())
    assert weight == 2
    torch.testing.assert_close(model.inference(audio, lengths)["encoded_feat"], audio)


@pytest.mark.parametrize("method", ["forward", "inference"])
@pytest.mark.parametrize("reference", ["ref_audio", "ref_text"])
def test_reference_requires_lengths(make_model, method, reference):
    from espnet2.universa.espnet_model import ESPnetUniversaModel

    model = ESPnetUniversaModel(make_model(), None)
    inputs = {
        "audio": torch.randn(2, 10, 8),
        "audio_lengths": torch.tensor([10, 8]),
        reference: (
            torch.zeros(2, 10, 8)
            if reference == "ref_audio"
            else torch.ones(2, 3, dtype=torch.long)
        ),
    }
    if method == "forward":
        inputs["metrics"] = {"mos": torch.ones(2)}
    with pytest.raises(ValueError, match=f"{reference}_lengths"):
        getattr(model, method)(**inputs)
