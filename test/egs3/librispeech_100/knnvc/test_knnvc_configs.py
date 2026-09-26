"""Tests that the recipe configs override the VC TEMPLATE defaults correctly."""

from pathlib import Path

from espnet3.utils.config_utils import load_and_merge_config

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_recipe_training_config_merges_over_template | Model, GAN optimizers,  |
# |                          | encoder and the official split survive the merge.|
# | test_recipe_publication_config_keeps_features_out_of_the_bundle |          |
# |                          | The published checkpoint is one pack_model keeps. |
# | test_recipe_metrics_config_scores_intelligibility |                     |
# |                          | measure has a real metric reading wav + ref.    |
# | test_pretrained_inference_config_uses_a_released_generator |               |
# |                          | inference_pretrained.yaml points at one of the    |
# |                          | authors' two released generators, not a `do_` state.|

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_recipe_training_config_merges_over_template(tmp_path: Path) -> None:
    recipe_config = _REPO_ROOT / "egs3/librispeech_100/knnvc/conf/training.yaml"
    cfg = load_and_merge_config(
        recipe_config, "training.yaml", default_package="egs3.TEMPLATE.knnvc"
    )
    assert cfg.model._target_ == "espnet3.systems.knnvc.vocoder.KNNVCVocoderModel"
    assert set(cfg.optimizers) == {"generator", "discriminator"}
    assert set(cfg.schedulers) == {"generator", "discriminator"}
    assert cfg.optimizers.generator.params == "generator"
    assert (
        cfg.prepare_features.encoder._target_
        == "espnet3.systems.knnvc.wavlm_encoder.WavLMEncoder"
    )
    assert cfg.prepare_features.encoder.layer == 6
    # The vocoder datasets read from the prepare_features output directory.
    train_args = cfg.dataset.train[0].data_src_args
    assert train_args.features_dir == cfg.prepare_features.features_dir
    assert train_args.kind == "vocoder" and train_args.split == "train-clean-100"
    valid_args = cfg.dataset.valid[0].data_src_args
    assert valid_args.split == "dev-clean" and valid_args.segment_frames is None
    # Both splits must get features, as in the official prematch script.
    assert [d.name for d in cfg.prepare_features.dataset] == [
        "train-clean-100",
        "dev-clean",
    ]


def test_recipe_publication_config_keeps_features_out_of_the_bundle() -> None:
    cfg = load_and_merge_config(
        _REPO_ROOT / "egs3/librispeech_100/knnvc/conf/publication.yaml",
        "publication.yaml",
        default_package="egs3.TEMPLATE.knnvc",
    )
    excludes = list(cfg.pack_model.exclude)
    assert "wavlm_l*_prematched" in excludes
    assert "last.ckpt" in excludes
    assert cfg.pack_model.readme.endswith("hf_model_readme.md")
    # The published checkpoint must be one the bundle actually contains.
    inference = load_and_merge_config(
        _REPO_ROOT / "egs3/librispeech_100/knnvc/conf/inference.yaml",
        "inference.yaml",
        default_package="egs3.TEMPLATE.knnvc",
    )
    checkpoint = inference.model.vocoder_checkpoint
    assert checkpoint.endswith(".pth")
    assert not any(Path(checkpoint).name == pattern for pattern in excludes)


def test_pretrained_inference_config_uses_a_released_generator() -> None:
    """The no-training config points at a released *generator*.

    The kNN-VC v0.1 release ships two generators (`prematch_g_02500000.pt`,
    trained on prematched features, and `g_02500000.pt`, trained on plain WavLM
    features) plus two `*_do_*.pt` discriminator/optimizer states that cannot be
    used for inference. Only the generators are valid here.
    """
    cfg = load_and_merge_config(
        _REPO_ROOT / "egs3/librispeech_100/knnvc/conf/inference_pretrained.yaml",
        "inference.yaml",
        default_package="egs3.TEMPLATE.knnvc",
    )
    checkpoint = cfg.model.vocoder_checkpoint
    assert checkpoint.startswith(
        "https://github.com/bshall/knn-vc/releases/download/v0.1/"
    )
    assert Path(checkpoint).name in ("prematch_g_02500000.pt", "g_02500000.pt")
    assert "_do_" not in Path(checkpoint).name


def test_recipe_metrics_config_scores_intelligibility() -> None:
    """``measure`` must have a metric to run, reading what ``infer`` writes."""
    cfg = load_and_merge_config(
        _REPO_ROOT / "egs3/librispeech_100/knnvc/conf/metrics.yaml",
        "metrics.yaml",
        default_package="egs3.TEMPLATE.knnvc",
    )
    assert cfg.metrics, "the recipe ships no metric, so measure cannot run"
    entry = cfg.metrics[0]
    assert entry.metric._target_.endswith("intelligibility.ASRIntelligibility")
    assert entry.metric.asr, "the ASR used for scoring must be named"
    # Both inputs are files the infer stage writes.
    assert dict(entry.inputs) == {"wav": "wav", "ref": "ref"}
