from egs3.TEMPLATE.vc.run import DEFAULT_STAGES
from espnet3.utils.config_utils import load_default_config


def test_default_stages_include_prepare_features_without_collect_stats() -> None:
    assert "prepare_features" in DEFAULT_STAGES
    assert "collect_stats" not in DEFAULT_STAGES
    assert "train_tokenizer" not in DEFAULT_STAGES
    assert DEFAULT_STAGES.index("prepare_features") < DEFAULT_STAGES.index("train")


def test_default_stages_omit_demo_stages() -> None:
    """VC ships no demo UI, so the demo stages stay out of the stage list."""
    assert "pack_demo" not in DEFAULT_STAGES
    assert "upload_demo" not in DEFAULT_STAGES
    assert DEFAULT_STAGES[-2:] == ["pack_model", "upload_model"]


def test_load_default_config_train_contains_expected_defaults() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.vc")
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert cfg.prepare_features.prematch is True
    assert cfg.prepare_features.topk == 4
    assert cfg.dataloader.train.iter_factory is None
    assert cfg.best_model_criterion[0][0] == "valid/mel_loss"


def test_load_default_config_infer_contains_expected_targets() -> None:
    cfg = load_default_config("inference.yaml", "egs3.TEMPLATE.vc")
    assert cfg.output_fn == "egs3.TEMPLATE.vc.src.inference.build_output"
    assert cfg.output_artifacts.wav.type == "wav"
    assert cfg.output_artifacts.wav.sample_rate == 16000
    assert (
        cfg.runner._target_ == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_build_output_wraps_waveform() -> None:
    from egs3.TEMPLATE.vc.src.inference import build_output

    wav = [0.0, 0.1]
    out = build_output(
        {"pair_id": "a_to_b", "text": "HELLO", "target_speaker": "b"}, wav, 0
    )
    assert out == {
        "utt_id": "a_to_b",
        "wav": wav,
        "ref": "HELLO",
        "target_speaker": "b",
    }
    assert build_output({}, wav, 3) == {"utt_id": "3", "wav": wav}
    batched = build_output([{"pair_id": "x"}, {"pair_id": "y"}], [wav, wav], [0, 1])
    assert [o["utt_id"] for o in batched] == ["x", "y"]
