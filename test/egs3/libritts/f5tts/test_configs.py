"""Guards that the shipped recipe configs stay loadable and consistent.

The recipe's configs are merged over ``egs3/TEMPLATE/f5tts/conf`` by the
template's ``run.py``; these tests load them the same way and pin the few
values the rest of the recipe depends on.
"""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from egs3.libritts.f5tts.src.inference import build_output
from espnet3.api.inference import Audio
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
    cfg = _load(monkeypatch, "inference.yaml", "inference.yaml")
    assert cfg.model._target_ == "espnet3.systems.f5tts.inference.Inference"
    assert list(cfg.input_key) == ["text", "reference_speech", "reference_text"]
    assert cfg.output_fn == "src.inference.build_output"


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


def test_build_output_keeps_the_metric_columns():
    """`ref` and `text` feed conf/metrics.yaml; `wav` is the Audio's samples."""
    output = build_output(
        {"utt_id": "u1", "raw_text": "hello", "ref_wav_path": "prompt.wav"},
        {"wav": Audio(np.zeros(4, dtype=np.float32), 24000)},
        0,
    )
    assert output["utt_id"] == "u1"
    assert output["text"] == "hello"
    assert output["ref"] == "prompt.wav"
    assert output["wav"].dtype == np.float32
    assert output["wav"].shape == (4,)


def test_build_output_falls_back_to_the_ground_truth_wav():
    output = build_output(
        {"raw_text": "hello", "wav_path": "gt.wav"},
        {"wav": np.zeros(2, dtype=np.float32)},
        3,
    )
    assert output["utt_id"] == "3"
    assert output["ref"] == "gt.wav"


def test_build_output_handles_a_batch():
    outputs = build_output(
        [{"utt_id": "a", "raw_text": "x"}, {"utt_id": "b", "raw_text": "y"}],
        [{"wav": np.zeros(1)}, {"wav": np.ones(1)}],
        [0, 1],
    )
    assert [output["utt_id"] for output in outputs] == ["a", "b"]
    assert outputs[1]["wav"].tolist() == [1.0]


def test_build_output_requires_a_waveform():
    with pytest.raises(RuntimeError, match="wav"):
        build_output({"utt_id": "a"}, {}, 0)
