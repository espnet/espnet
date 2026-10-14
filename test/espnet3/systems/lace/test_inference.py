"""Unit tests for espnet3.systems.lace.inference."""

from test.espnet3.systems.lace._codec_dummies import (
    HOP_LENGTH,
    NUM_CODEBOOKS,
    write_tiny_bundle,
)

import numpy as np
import pytest
import torch
import yaml

from espnet3.api.inference import Audio, load
from espnet3.publication.schema import PACK_SCHEMA_VERSION
from espnet3.systems.lace.inference import Inference, LACEAudioCoding
from espnet3.systems.lace.src.lace import LACE
from espnet3.systems.lace.system import (
    build_lace_model_config,
    write_lace_model_config,
)

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


# ---------------- Inference (espnet3.api.inference contract) ----------------


def test_inference_declaration_and_sample_rate(finetuned):
    """``Inference`` declares one audio in, one audio out, at the codec rate."""
    spec_path, model_file = finetuned
    model = Inference(train_config=spec_path, model_file=model_file, rate=0.5)
    assert [f.name for f in model.inputs] == ["audio"]
    assert [f.name for f in model.outputs] == ["resyn_audio"]
    assert model.sample_rate == 120  # TINY_ENCODEC_CONF sampling_rate
    assert isinstance(model.backend, LACEAudioCoding)


def test_inference_run_matches_the_backend_roundtrip(finetuned):
    """``Inference`` returns exactly the backend's resynthesis as ``Audio``."""
    spec_path, model_file = finetuned
    model = Inference(train_config=spec_path, model_file=model_file, rate=0.5)
    audio = np.random.RandomState(0).randn(240).astype(np.float32)

    output = model(audio)

    assert isinstance(output["resyn_audio"], Audio)
    assert output["resyn_audio"].rate == model.sample_rate
    expected = model.backend(torch.as_tensor(audio))["resyn_audio"].numpy()
    assert np.array_equal(output["resyn_audio"].array, expected)
    # The same model behind a (rate, samples) pair and a batch of items.
    assert model((120, audio))["resyn_audio"].array.shape == expected.shape
    assert len(model.batch([{"audio": audio}, {"audio": audio}])) == 2


def test_inference_wraps_a_built_backend(finetuned):
    """An already built ``LACEAudioCoding`` is accepted, without extra kwargs."""
    spec_path, model_file = finetuned
    coding = LACEAudioCoding(train_config=spec_path, model_file=model_file)
    assert Inference(coding).backend is coding
    with pytest.raises(TypeError, match="built one was too"):
        Inference(coding, rate=0.5)


def test_inference_from_pretrained_bundle(finetuned, tmp_path):
    """A pack_model-shaped bundle whose inference.yaml names this Inference."""
    spec_path, model_file = finetuned
    pack = tmp_path / "model_pack"
    (pack / "conf").mkdir(parents=True)
    (pack / "conf" / "inference.yaml").write_text(
        yaml.safe_dump(
            {
                "recipe_dir": ".",
                "model": {
                    "_target_": "espnet3.systems.lace.inference.Inference",
                    "train_config": spec_path,
                    "model_file": model_file,
                    "rate": 0.5,
                },
            }
        )
    )
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

    model = Inference.from_pretrained(pack)
    assert isinstance(model, Inference)
    assert model.backend.rate == 0.5
    # Any backend argument replaces the packed one, as a rate sweep needs.
    assert Inference.from_pretrained(pack, rate=0.25).backend.rate == 0.25
    # And the generic loader finds the class through meta.yaml's `system`.
    assert isinstance(load(pack), Inference)
