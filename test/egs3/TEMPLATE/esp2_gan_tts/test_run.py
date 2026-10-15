"""Tests for the egs3/TEMPLATE/esp2_gan_tts defaults and runner wiring."""

from pathlib import Path

from egs3.TEMPLATE.esp2_gan_tts.run import DEFAULT_STAGES
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

PACKAGE = "egs3.TEMPLATE.esp2_gan_tts"


def test_default_stages_run_manifest_consumers_after_create_dataset() -> None:
    """Manifest-consuming stages run after create_dataset, before collect_stats.

    Execution always follows list order, whatever order ``--stages`` is
    given in, so ``--stages all`` must not reach compute_xvectors first.
    """
    order = {stage: idx for idx, stage in enumerate(DEFAULT_STAGES)}
    assert order["create_dataset"] < order["compute_xvectors"]
    assert order["compute_xvectors"] < order["remove_long_short"]
    assert order["remove_long_short"] < order["create_token_list"]
    assert order["create_token_list"] < order["collect_stats"]
    assert order["collect_stats"] < order["train"] < order["infer"] < order["measure"]


def test_load_default_config_train_contains_expected_targets() -> None:
    cfg = load_default_config("training.yaml", PACKAGE)
    assert cfg.task == "espnet3.systems.esp2_gan_tts.task.GANTTSTask"
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert cfg.dataset._recursive_ is False
    assert cfg.dataset.recipe_dir == "."
    # The preprocessor never carries `train`: DataOrganizer sets it per split.
    assert "train" not in cfg.dataset.preprocessor
    # Stage defaults Masao asked to live in the template.
    assert cfg.remove_long_short.min_wav_duration == 1.0
    assert cfg.remove_long_short.max_wav_duration == 20.0
    assert cfg.xvector.toolkit == "espnet"
    assert cfg.xvector.pretrained_model == "espnet/voxcelebs12_rawnet3"
    assert cfg.create_token_list.token_type == "phn"
    assert cfg.trainer.gan.generator_first is False


def test_load_default_config_infer_contains_expected_targets() -> None:
    cfg = load_default_config("inference.yaml", PACKAGE)
    assert cfg.dataset.recipe_dir == "."
    assert cfg.dataset._recursive_ is False
    assert [entry.name for entry in cfg.dataset.test] == ["valid", "test"]
    assert "train" not in cfg.dataset.preprocessor
    assert cfg.model._target_ == "espnet2.bin.tts_inference.Text2Speech"
    assert (
        cfg.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        cfg.runner._target_ == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_load_default_config_metrics_uses_versa() -> None:
    cfg = load_default_config("metrics.yaml", PACKAGE)
    assert [entry.name for entry in cfg.dataset.test] == ["valid", "test"]
    metric = cfg.metrics[0].metric
    assert metric._target_ == "espnet3.components.metrics.versa.VersaMetric"
    assert [score.name for score in metric.score_config] == [
        "pseudo_mos",
        "whisper_wer",
        "speaker",
    ]
    assert (metric.wav_key, metric.ref_key, metric.text_key) == ("wav", "ref", "text")


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path: Path) -> None:
    user = tmp_path / "train_user.yaml"
    user.write_text(
        """
exp_tag: user_train
xvector:
  toolkit: speechbrain
  pretrained_model: speechbrain/spkrec-ecapa-voxceleb
create_token_list:
  filename: phn_tokens.txt
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_and_merge_config(user, "training.yaml", default_package=PACKAGE)

    # user config overrides the template defaults ...
    assert cfg.xvector.toolkit == "speechbrain"
    assert cfg.create_token_list.filename == "phn_tokens.txt"
    # ... while untouched template defaults and interpolations survive.
    assert cfg.xvector.splits == ["train", "valid", "test"]
    assert cfg.remove_long_short.max_wav_duration == 20.0
    assert cfg.dataset.preprocessor.token_list.endswith("tokens/phn_tokens.txt")
    assert cfg.dataset.preprocessor.text_cleaner == ["tacotron"]
