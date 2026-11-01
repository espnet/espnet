"""Unit tests for espnet3.systems.lace.src.inference."""

from test.espnet3.systems.lace._codec_dummies import (
    HOP_LENGTH,
    NUM_CODEBOOKS,
    write_tiny_bundle,
)

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from espnet3.contract.kinds.audio import Audio
from espnet3.publication import InferenceModel
from espnet3.publication.schema import PACK_SCHEMA_VERSION
from espnet3.systems.lace.src.inference import LACEAudioCoding
from espnet3.systems.lace.src.lace import LACE
from espnet3.systems.lace.system import (
    build_lace_model_config,
    write_lace_model_config,
)
from espnet3.utils.config_utils import load_default_config

COMPRESSION = {"name": "cosine_similarity", "params": {}, "eval_rate": 0.5}


@pytest.fixture
def finetuned(tmp_path):
    """Fine-tuning run: lace_model.yaml spec + wrapped-layout checkpoint."""
    write_tiny_bundle(tmp_path / "bundle", lightning=True)
    model_config = {
        "_target_": "espnet3.systems.lace.src.lace.LACE",
        "compression_model": COMPRESSION,
        "pretrained_model_dir": str(tmp_path / "bundle"),
    }
    spec_path = write_lace_model_config(model_config, tmp_path / "lace_model.yaml")
    model = LACE(**build_lace_model_config(model_config))
    model_file = tmp_path / "finetuned.ckpt"
    torch.save({"state_dict": model.state_dict()}, model_file)
    return str(spec_path), str(model_file)


def test_mismatched_checkpoint_raises(finetuned, tmp_path):
    """A checkpoint that does not match the rebuilt model raises."""
    spec_path, _ = finetuned
    wrong = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": {"unexpected.weight": torch.zeros(1)}}, wrong)
    with pytest.raises(RuntimeError):
        LACEAudioCoding(train_config=spec_path, model_file=str(wrong))


def test_roundtrip_with_finetuned_checkpoint(finetuned):
    """The roundtrip returns codes and audio for 1-D and batched input."""
    spec_path, model_file = finetuned
    coding = LACEAudioCoding(train_config=spec_path, model_file=model_file, rate=0.5)
    assert not coding.model.training
    assert coding.sample_rate == coding.fs == 120  # TINY_ENCODEC_CONF sampling_rate
    audio = torch.randn(240)
    output = coding(audio)
    assert sorted(output) == ["codes", "resyn_audio"]
    assert output["codes"].shape == (NUM_CODEBOOKS, 1, 240 // HOP_LENGTH)
    assert output["resyn_audio"].shape == (240,)
    assert sorted(coding(audio.numpy(), encode_only=True)) == ["codes"]
    batch = coding(np.zeros((2, 1, 240), dtype=np.float32))
    assert batch["resyn_audio"].shape == (2, 1, 240)
    with pytest.raises(AssertionError):
        coding(None)


def test_inference_state_is_reset_after_each_call(finetuned):
    """The quantizer's inference rate and anchor are set per call and reset."""
    spec_path, model_file = finetuned
    coding = LACEAudioCoding(
        train_config=spec_path, model_file=model_file, rate=0.5, anchor_start_layer=1
    )
    seen = {}
    original_encode = coding.model.encode

    def spy_encode(audio, **kwargs):
        """Record the quantizer's inference state at encode time, then encode."""
        seen["rate"] = coding.quantizer._inference_rate
        seen["anchor"] = coding.quantizer._inference_anchor_start_layer
        seen["kwargs"] = kwargs
        return original_encode(audio, **kwargs)

    coding.model.encode = spy_encode
    coding(torch.randn(240))
    assert seen == {"rate": 0.5, "anchor": 1, "kwargs": {"target_bw": None}}
    coding(torch.randn(240), decode_conf={"rate": 0.25, "anchor_start_layer": None})
    assert (seen["rate"], seen["anchor"]) == (0.25, None)
    assert coding.quantizer._inference_rate is None
    assert coding.quantizer._inference_anchor_start_layer is None


# ---------------- packed bundle through the publication loader ----------------


def _write_pack(pack, spec_path, model_file):
    """Write a pack_model-shaped bundle whose inference.yaml is the template's.

    The packed config is what `pack_model` writes: the template defaults (the
    `LACEAudioCoding` target, `input_key`, the contract and the runner) with
    the recipe's model arguments, and `meta.yaml` pointing at it.
    """
    config = OmegaConf.to_container(
        load_default_config("inference.yaml", "egs3.TEMPLATE.lace"), resolve=False
    )
    packed = {
        "recipe_dir": ".",
        "model": {
            "_target_": config["model"]["_target_"],
            "train_config": spec_path,
            "model_file": model_file,
            "rate": 0.5,
        },
        "input_key": config["input_key"],
        "contract": config["contract"],
        "provider": config["provider"],
        "runner": config["runner"],
    }
    (pack / "conf").mkdir(parents=True)
    (pack / "conf" / "inference.yaml").write_text(yaml.safe_dump(packed))
    (pack / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": PACK_SCHEMA_VERSION,
                "system": "lace",
                "files": {},
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            }
        )
    )


def test_packed_bundle_loads_untrusted_and_matches_the_backend(finetuned, tmp_path):
    """`InferenceModel.from_packed` runs the template's contract around the codec.

    The bundle names only installed classes, so it loads with the default
    `trust_user_code=False`; the output is an `Audio` at the codec's own rate
    (nothing in the contract says 24000) and equals the backend's roundtrip.
    """
    spec_path, model_file = finetuned
    pack = tmp_path / "model_pack"
    _write_pack(pack, spec_path, model_file)
    audio = np.random.RandomState(0).randn(240).astype(np.float32)

    model = InferenceModel.from_packed(pack)
    output = model(audio)

    assert isinstance(model.model, LACEAudioCoding)
    assert isinstance(output["resyn_audio"], Audio)
    assert output["resyn_audio"].rate == 120
    expected = model.model(torch.as_tensor(audio))["resyn_audio"].numpy()
    assert np.array_equal(output["resyn_audio"].array, expected)
    # A (rate, samples) pair at another rate is resampled to the codec's.
    assert model((240, np.zeros(480, dtype=np.float32)))["resyn_audio"].rate == 120


def test_packed_bundle_takes_backend_overrides(finetuned, tmp_path):
    """Any `LACEAudioCoding` argument replaces the packed one, as a rate sweep needs."""
    spec_path, model_file = finetuned
    pack = tmp_path / "model_pack"
    _write_pack(pack, spec_path, model_file)

    assert InferenceModel.from_packed(pack).model.rate == 0.5
    assert InferenceModel.from_packed(pack, rate=0.25).model.rate == 0.25
