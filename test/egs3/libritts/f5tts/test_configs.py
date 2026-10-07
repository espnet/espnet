"""Guards that the shipped recipe configs stay loadable and consistent.

The recipe's configs are merged over ``egs3/TEMPLATE/f5tts/conf`` by the
template's ``run.py``; these tests load them the same way and pin the few
values the rest of the recipe depends on.
"""

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[4] / "egs3" / "libritts" / "f5tts"
TEMPLATE_PACKAGE = "egs3.TEMPLATE.f5tts"


def _load(monkeypatch, name, config_name):
    """Load a recipe config the way the template's run.py does."""
    monkeypatch.chdir(RECIPE)
    return load_and_merge_config(
        Path("conf") / name,
        config_name=config_name,
        default_package=TEMPLATE_PACKAGE,
        resolve=False,
    )


def _raw(name):
    """Return the config as plain dicts with interpolations left unresolved.

    OmegaConf resolves interpolations lazily on attribute access, so reading
    ``cfg.model.train_config`` off a loaded config would hand back the
    resolved path rather than the literal ``${recipe_dir}/...`` string these
    tests assert on.
    """
    return OmegaConf.to_container(OmegaConf.load(RECIPE / "conf" / name), resolve=False)


def test_training_config_loads_over_the_template(monkeypatch):
    cfg = _load(monkeypatch, "training.yaml", "training.yaml")
    assert cfg.model._target_ == "espnet3.systems.f5tts.f5tts.F5TTS"
    assert cfg.task is None
    # The template's TensorBoard logger is kept, not replaced by a hosted one.
    assert [entry["_target_"] for entry in cfg.trainer.logger] == [
        "lightning.pytorch.loggers.TensorBoardLogger"
    ]


def test_training_config_has_one_token_list_path():
    """The preprocessor, the model and create_token_list agree on the file."""
    cfg = _raw("training.yaml")
    assert cfg["token_list"] == (
        "${create_token_list.save_path}/${create_token_list.filename}"
    )
    assert cfg["model"]["token_list"] == "${token_list}"
    assert cfg["dataset"]["preprocessor"]["token_list"] == "${token_list}"


def test_inference_config_loads(monkeypatch):
    """The model is the system's ``Inference``, so the declaration drives infer.

    ``input_key`` and ``output_fn`` are refused next to an ``Inference``, and
    ``output_artifacts`` is unnecessary: the runner writes ``wav`` as audio.
    """
    cfg = _load(monkeypatch, "inference.yaml", "inference.yaml")
    assert cfg.model._target_ == "espnet3.systems.f5tts.inference.Inference"
    assert cfg.get("input_key") is None
    assert cfg.get("output_fn") is None
    assert cfg.get("output_artifacts") is None


def test_default_inference_config_uses_librispeech_pc():
    test_sets = _raw("inference.yaml")["dataset"]["test"]
    assert len(test_sets) == 1
    assert test_sets[0]["name"] == "librispeech_pc"
    assert test_sets[0]["data_src"] == "egs3.libritts.f5tts.dataset.librispeech_pc"
    assert test_sets[0]["data_src_args"]["fs"] == 24000


def test_default_inference_config_is_portable():
    cfg = _raw("inference.yaml")
    assert cfg["model"]["train_config"] == "${recipe_dir}/conf/training.yaml"
    assert cfg["model"]["checkpoint_path"] == "${exp_dir}/last.ckpt"
    # Empty exp_tag means this config is training-backed: run.py must be given
    # --training_config alongside it.
    assert not cfg["exp_tag"]


def test_inference_config_train_config_exists():
    """The inference config must point at a training config that is present.

    ``--training_config`` never overrides an inference config's own
    ``model.train_config``, so a stale value here is not caught at the CLI:
    it surfaces as a checkpoint shape mismatch part way into a GPU job.
    """
    train_config = _raw("inference.yaml")["model"]["train_config"]
    assert train_config.startswith("${recipe_dir}/")
    resolved = RECIPE / train_config.removeprefix("${recipe_dir}/")
    assert resolved.is_file(), f"inference.yaml points at missing {train_config}"


def test_metrics_config_matches_official_protocol():
    cfg = _raw("metrics.yaml")
    assert [entry["name"] for entry in cfg["dataset"]["test"]] == ["librispeech_pc"]

    score_config = cfg["metrics"][0]["metric"]["score_config"]
    by_name = {entry["name"]: entry for entry in score_config}

    # WER: faster-whisper large-v3, beam 5, float16, engine-identical to the
    # official eval_librispeech_test_clean.py.
    wer = by_name["fwhisper_wer"]
    assert wer["model_tag"] == "large-v3"
    assert wer["beam_size"] == 5
    assert wer["compute_type"] == "float16"
    assert wer["text_cleaner"] == "whisper_basic"

    # UTMOS only: dnsmos is not part of the official protocol.
    assert by_name["pseudo_mos"]["predictor_types"] == ["utmos"]

    # SIM: documented deviation, the nearest ESPnet-SPK model.
    assert by_name["speaker"]["model_tag"] == "espnet/voxcelebs12_ecapa_wavlm_joint"


def test_metrics_config_reads_the_references_from_the_data():
    """infer writes only `wav.scp`; the prompt wav and target text come from the data.

    `ref_wav_path` is the pinned prompt, which makes the speaker similarity
    generated-vs-prompt (SIM-o) by construction.
    """
    inputs = _raw("metrics.yaml")["metrics"][0]["inputs"]
    assert inputs == {
        "wav": "wav",
        "ref": "dataset:ref_wav_path",
        "text": "dataset:text",
    }
