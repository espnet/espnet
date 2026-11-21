"""Tests for the egs3/TEMPLATE/esp2_gan_tts defaults and launcher wiring."""

from pathlib import Path

from espnet3.systems.esp2_gan_tts.system import ESP2GANTTSSystem
from espnet3.systems.launch import default_conf_package
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

PACKAGE = "egs3.TEMPLATE.esp2_gan_tts"
RECIPE_CONF = Path(__file__).resolve().parents[4] / "egs3/libritts/esp2_gan_tts/conf"


def test_stages_run_manifest_consumers_after_create_dataset() -> None:
    """Manifest-consuming stages run after create_dataset, before collect_stats.

    Execution always follows the declared order, whatever order ``--stages``
    is given in, so ``--stages all`` must not reach compute_xvectors first.
    """
    order = {spec.name: idx for idx, spec in enumerate(ESP2GANTTSSystem.stages)}
    assert order["create_dataset"] < order["compute_xvectors"]
    assert order["compute_xvectors"] < order["remove_long_short"]
    assert order["remove_long_short"] < order["create_token_list"]
    assert order["create_token_list"] < order["collect_stats"]
    assert order["collect_stats"] < order["train"] < order["infer"] < order["measure"]


def test_launch_reads_the_defaults_from_this_template() -> None:
    """`launch(ESP2GANTTSSystem)` merges recipe configs over this package."""
    assert default_conf_package(ESP2GANTTSSystem) == PACKAGE


def test_load_default_config_train_contains_expected_targets() -> None:
    """The training defaults name the GAN-TTS task and the data organizer."""
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
    # Text2Speech rebuilds its text preprocessing from `args.cleaner` /
    # `args.g2p` of the written config.yaml, so the model block carries them.
    assert cfg.model.cleaner == "tacotron"
    assert cfg.model.g2p == "espeak_ng_english_us_vits"


def test_load_default_config_infer_declares_the_contract() -> None:
    """The inference defaults declare text + spembs -> wav, with no preprocessor."""
    cfg = load_default_config("inference.yaml", PACKAGE)
    assert cfg.dataset.recipe_dir == "."
    assert cfg.dataset._recursive_ is False
    assert [entry.name for entry in cfg.dataset.test] == ["valid", "test"]
    # Raw text goes to Text2Speech, which tokenizes it itself; the dataset
    # serves the references and skips the waveform.
    assert cfg.dataset.preprocessor is None
    for entry in cfg.dataset.test:
        assert entry.data_src_args.load_speech is False
        assert entry.data_src_args.inference is True
    assert cfg.model._target_ == "espnet2.bin.tts_inference.Text2Speech"
    assert cfg.input_key == ["text", "spembs"]
    assert cfg.batch_size is None
    assert list(cfg.contract.inputs) == ["text", "spembs"]
    assert cfg.contract.inputs.text.kind == "text"
    assert cfg.contract.inputs.spembs.kind == "vector"
    assert list(cfg.contract.outputs) == ["wav"]
    assert cfg.contract.outputs.wav.kind == "audio"
    assert cfg.contract.outputs.wav.sample_rate == 22050
    assert cfg.contract.outputs.wav.channels == 1
    assert "output_fn" not in cfg and "output_artifacts" not in cfg
    assert (
        cfg.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        cfg.runner._target_
        == "espnet3.systems.esp2_gan_tts.src.inference_runner.TTSInferenceRunner"
    )


def test_load_default_config_metrics_binds_versa_to_the_dataset_columns() -> None:
    """The metrics defaults bind VersaMetric's references to dataset columns."""
    cfg = load_default_config("metrics.yaml", PACKAGE)
    assert [entry.name for entry in cfg.dataset.test] == ["valid", "test"]
    entry = cfg.metrics[0]
    metric = entry.metric
    assert metric._target_ == "espnet3.systems.esp2_gan_tts.metrics.versa.VersaMetric"
    assert [score.name for score in metric.score_config] == [
        "pseudo_mos",
        "whisper_wer",
        "speaker",
    ]
    assert list(metric.outputs) == [
        "whisper_wer",
        "whisper_cer",
        "spk_similarity",
        "utmos",
        "dns_overall",
        "dns_p808",
    ]
    assert (metric.wav_key, metric.ref_key, metric.text_key) == ("wav", "ref", "text")
    assert dict(entry.inputs) == {
        "wav": "wav",
        "ref": "dataset:wav_path",
        "text": "dataset:raw_text",
    }


def test_load_default_config_publication_ships_the_checkpoint_and_tokens() -> None:
    """The publication defaults keep last.ckpt and the token list, no code."""
    cfg = load_default_config("publication.yaml", PACKAGE)
    assert [str(p) for p in cfg.pack_model.include] == ["./data/tokens"]
    # The packed inference config loads `${exp_dir}/last.ckpt`, so the bulk
    # copy must keep it; no recipe code is shipped.
    assert "last.ckpt" not in cfg.pack_model.exclude
    assert "src" not in cfg.pack_model.include
    assert "dataset" not in cfg.pack_model.include


def test_load_default_config_demo_wires_the_reference_recording() -> None:
    """The demo shows text + a reference recording and reuses the training extractor."""
    cfg = load_default_config("demo.yaml", PACKAGE)
    assert cfg.model.trust_user_code is False
    assert cfg.ui.app_script.endswith("egs3/TEMPLATE/esp2_gan_tts/src/app.py")
    assert cfg.ui.description == "egs3/TEMPLATE/esp2_gan_tts/src/demo_description.md"
    assert [spec.key for spec in cfg.ui.inputs] == ["text", "reference_speech"]
    assert [spec.type for spec in cfg.ui.inputs] == ["text", "audio"]
    assert "outputs" not in cfg.ui
    # Same extractor as the training template's compute_xvectors defaults.
    training = load_default_config("training.yaml", PACKAGE)
    assert cfg.xvector.toolkit == training.xvector.toolkit
    assert cfg.xvector.pretrained_model == training.xvector.pretrained_model


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path: Path) -> None:
    """A user config merged over the template overrides only the keys it sets."""
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
    assert cfg.model.cleaner == "tacotron"


def test_libritts_recipe_overlays_hold_only_their_deltas() -> None:
    """The recipe inherits the contract, runner and launcher from the template."""
    inference = load_and_merge_config(
        RECIPE_CONF / "inference.yaml", "inference.yaml", default_package=PACKAGE
    )
    assert inference.xvector.spk_embed_tag == "spkrec-ecapa-voxceleb"
    assert inference.dataset.preprocessor is None
    assert inference.input_key == ["text", "spembs"]
    assert list(inference.contract.inputs) == ["text", "spembs"]
    assert inference.dataset.test[0].data_src_args.xvector_dir.endswith(
        "spkrec-ecapa-voxceleb_valid"
    )

    demo = load_and_merge_config(
        RECIPE_CONF / "demo.yaml", "demo.yaml", default_package=PACKAGE, resolve=False
    )
    assert demo.xvector.toolkit == "speechbrain"
    assert demo.xvector.pretrained_model == "speechbrain/spkrec-ecapa-voxceleb"
    assert "speechbrain" in demo.pack.requirements
    assert demo.ui.app_script.endswith("egs3/TEMPLATE/esp2_gan_tts/src/app.py")
    assert demo.model.trust_user_code is False

    publication = load_and_merge_config(
        RECIPE_CONF / "publication.yaml", "publication.yaml", default_package=PACKAGE
    )
    assert [str(p) for p in publication.pack_model.include] == ["./data/tokens"]


def test_load_and_merge_config_none_path_returns_none() -> None:
    """A missing user config path yields None rather than the defaults."""
    assert load_and_merge_config(None, "metrics.yaml", default_package=PACKAGE) is None
