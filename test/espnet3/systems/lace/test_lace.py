"""Unit tests for espnet3.systems.lace.src.lace."""

from test.espnet3.systems.lace._codec_dummies import (
    build_tiny_codec_conf,
    build_tiny_codec_model,
    write_tiny_bundle,
)

import pytest
import torch
from omegaconf import OmegaConf

from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.systems.lace.src.dp_segmentation import DPSegmentationCompression
from espnet3.systems.lace.src.identity import IdentityCompression
from espnet3.systems.lace.src.lace import (
    COMPRESSION_MODEL_MAP,
    LACE,
    build_codec_from_bundle,
    build_compression_model,
    drop_legacy_dac_discriminator_keys,
    fill_missing_snake_alpha,
    freeze_codec_module,
    load_model_state_strict,
    read_bundle_artifacts,
    read_model_state,
)
from espnet3.systems.lace.src.quantizer import CompressionResidualVectorQuantizer

COMPRESSION = {"name": "cosine_similarity", "params": {}, "eval_rate": 0.5}


def test_build_compression_model():
    """Names map to classes with their kwargs; unknown names raise."""
    model = build_compression_model("dp_segmentation", max_span=3)
    assert isinstance(model, DPSegmentationCompression)
    assert model.max_span == 3
    assert isinstance(build_compression_model("none"), IdentityCompression)
    assert set(COMPRESSION_MODEL_MAP) >= {"cosine_similarity", "density_peak"}
    with pytest.raises(ValueError, match="Unknown compression model"):
        build_compression_model("unknown")


def test_build_from_fresh_codec_conf_with_omegaconf_inputs():
    """A fresh codec builds from OmegaConf inputs and keeps the espnet2 layout."""
    model = LACE(
        compression_model=OmegaConf.create(COMPRESSION),
        codec="encodec",
        codec_conf=OmegaConf.create(build_tiny_codec_conf()),
    )
    assert isinstance(model, AbsGANESPnetModel)
    assert isinstance(model, ESPnetGANCodecModel)
    # The espnet2 state-dict layout is kept, with the wrapper under `.rvq`.
    assert all(k.startswith("codec.") for k in model.state_dict())
    assert any(
        k.startswith("codec.generator.quantizer.rvq.") for k in model.state_dict()
    )
    quantizer = model.codec.generator.quantizer
    assert isinstance(quantizer, CompressionResidualVectorQuantizer)
    assert quantizer.eval_rate == 0.5
    output = model(audio=torch.randn(2, 240), forward_generator=True)
    assert torch.isfinite(output["loss"])


@pytest.mark.parametrize("layout", ["espnet2", "espnet3"])
@pytest.mark.parametrize("lightning", [False, True])
def test_pretrained_model_dir_loads_bundle_weights_before_wrapping(
    tmp_path, layout, lightning
):
    """Bundle weights land under the wrapped keys for both layouts."""
    baseline = write_tiny_bundle(tmp_path, layout=layout, lightning=lightning)
    model = LACE(compression_model=COMPRESSION, pretrained_model_dir=str(tmp_path))
    assert isinstance(
        model.codec.generator.quantizer, CompressionResidualVectorQuantizer
    )
    state_dict = model.state_dict()
    for key, value in baseline.state_dict().items():
        wrapped_key = key.replace("quantizer.vq.", "quantizer.rvq.vq.rvq.")
        assert torch.equal(state_dict[wrapped_key], value), key


def test_pretrained_model_tag_downloads_through_espnet_model_zoo(tmp_path, monkeypatch):
    """A tag goes through ``ModelDownloader`` and its weights are loaded."""
    from espnet_model_zoo.downloader import ModelDownloader

    baseline = write_tiny_bundle(tmp_path)
    artifacts = read_bundle_artifacts(tmp_path)
    calls = {}

    def download_and_unpack(self, name):
        calls["name"] = name
        return artifacts

    monkeypatch.setattr(ModelDownloader, "download_and_unpack", download_and_unpack)
    model = LACE(
        compression_model=COMPRESSION,
        pretrained_model_tag="espnet/libritts_encodec_24k",
    )
    assert calls == {"name": "espnet/libritts_encodec_24k"}
    encoder_state = model.codec.generator.encoder.state_dict()
    for key, value in baseline.codec.generator.encoder.state_dict().items():
        assert torch.equal(encoder_state[key], value), key


def test_read_bundle_artifacts(tmp_path):
    """``meta.yaml`` entries become absolute paths; a missing file raises."""
    write_tiny_bundle(tmp_path, layout="espnet3")
    artifacts = read_bundle_artifacts(tmp_path)
    assert artifacts == {
        "model_file": str(tmp_path / "exp" / "model.pth"),
        "training_config": str(tmp_path / "conf" / "training.yaml"),
    }
    with pytest.raises(FileNotFoundError, match="no meta.yaml"):
        read_bundle_artifacts(tmp_path / "missing")


def test_build_codec_from_bundle_rejects_unknown_layouts(tmp_path):
    """Artifacts without a model file or a known config key raise."""
    with pytest.raises(ValueError, match="no 'model_file'"):
        build_codec_from_bundle({"train_config": "x"})
    with pytest.raises(ValueError, match="neither 'train_config'"):
        build_codec_from_bundle({"model_file": "x", "inference_config": "y"})


def test_build_codec_from_bundle_strict_load_mismatch_raises(tmp_path):
    """A bundle whose weights miss a key raises instead of partial-loading."""
    write_tiny_bundle(tmp_path)
    artifacts = read_bundle_artifacts(tmp_path)
    broken = dict(build_tiny_codec_model().state_dict())
    broken.pop(next(iter(broken)))
    torch.save(broken, artifacts["model_file"])
    with pytest.raises(RuntimeError, match="Missing key"):
        build_codec_from_bundle(artifacts)


def test_architecture_source_validation(tmp_path):
    """Exactly one model source must be given."""
    with pytest.raises(ValueError, match="exactly one model source"):
        LACE(compression_model=COMPRESSION)
    with pytest.raises(ValueError, match="exactly one model source"):
        LACE(
            compression_model=COMPRESSION,
            codec="encodec",
            pretrained_model_dir=str(tmp_path),
        )
    with pytest.raises(ValueError, match="exactly one model source"):
        LACE(
            compression_model=COMPRESSION,
            pretrained_model_tag="espnet/libritts_encodec_24k",
            pretrained_model_dir=str(tmp_path),
        )


def test_rejects_non_residual_vector_quantizer(monkeypatch):
    """A codec without the shared RVQ is refused."""
    import espnet3.utils.task_utils as task_utils

    model = build_tiny_codec_model()
    model.codec.generator.quantizer = torch.nn.Identity()
    monkeypatch.setattr(task_utils, "get_espnet_model", lambda task, config: model)
    with pytest.raises(TypeError, match="ResidualVectorQuantizer"):
        LACE(compression_model=COMPRESSION, codec="encodec")


def test_load_model_state_strict_raises_on_mismatch(tmp_path):
    """The strict loader raises on a missing key."""
    model = build_tiny_codec_model()
    state_dict = dict(model.state_dict())
    state_dict.pop(next(iter(state_dict)))
    checkpoint_path = tmp_path / "broken.pth"
    torch.save(state_dict, checkpoint_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_state_strict(model, str(checkpoint_path))


@pytest.mark.parametrize("module", ["none", "encoder", "decoder"])
def test_freeze_codec_module(module):
    """Only the named generator module loses ``requires_grad``."""
    model = build_tiny_codec_model()
    freeze_codec_module(model, module)
    generator = model.codec.generator
    for name in ["encoder", "decoder"]:
        trainable = all(p.requires_grad for p in getattr(generator, name).parameters())
        assert trainable == (name != module)


def test_freeze_codec_module_rejects_unknown_choice():
    """``all`` is not a freeze choice."""
    with pytest.raises(ValueError, match="freeze_codec_module"):
        freeze_codec_module(build_tiny_codec_model(), "all")


# ---------------- legacy DAC bundles (published before espnet2's DAC changed) ----


def test_drop_legacy_dac_discriminator_keys(caplog):
    """Legacy DAC keys are dropped from a copy, with a warning, only for DAC."""
    legacy = {
        "codec": "dac",
        "codec_conf": {
            "discriminator_params": {
                "msmpmb_discriminator_params": {"periods": [2, 3]},
                "scales": 3,
                "scale_downsample_pooling": "AvgPool1d",
                "scale_downsample_pooling_params": {"kernel_size": 4},
            }
        },
    }
    with caplog.at_level("WARNING"):
        cleaned = drop_legacy_dac_discriminator_keys(legacy)
    assert cleaned["codec_conf"]["discriminator_params"] == {
        "msmpmb_discriminator_params": {"periods": [2, 3]}
    }
    assert "Dropping discriminator_params" in caplog.text
    assert "scales" in legacy["codec_conf"]["discriminator_params"]  # input untouched
    # Nothing to drop, or not DAC: the same object comes back, silently.
    caplog.clear()
    assert drop_legacy_dac_discriminator_keys(cleaned) is cleaned
    other = {"codec": "encodec", "codec_conf": {"discriminator_params": {"scales": 3}}}
    assert drop_legacy_dac_discriminator_keys(other) is other
    assert caplog.text == ""


def test_fill_missing_snake_alpha(caplog):
    """Missing Snake alphas are filled with ones; nothing else is added."""
    from espnet2.gan_codec.shared.encoder.snake_activation import Snake1d

    model = torch.nn.Sequential(Snake1d(), torch.nn.Linear(2, 2), Snake1d())
    with torch.no_grad():
        model[0].alpha.fill_(2.0)
    saved = {k: v for k, v in model.state_dict().items() if k != "2.alpha"}

    with caplog.at_level("WARNING"):
        filled = fill_missing_snake_alpha(model, saved)

    assert "Filling 1 Snake" in caplog.text
    assert "2.alpha" not in saved  # input untouched
    model.load_state_dict(filled, strict=True)
    assert model[0].alpha.item() == 2.0  # a saved alpha wins
    assert model[2].alpha.item() == 1.0  # the missing one is the trained value
    # A complete state dict comes back as it is.
    assert fill_missing_snake_alpha(model, filled) is filled
    # Only Snake alphas are filled, never an arbitrary missing key.
    partial = {k: v for k, v in filled.items() if k != "1.weight"}
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(fill_missing_snake_alpha(model, partial), strict=True)


def test_read_model_state_unwraps_lightning_and_model_prefix(tmp_path):
    """Plain, Lightning and ``model.``-prefixed checkpoints read the same."""
    model = build_tiny_codec_model()
    plain = tmp_path / "model.pth"
    torch.save(model.state_dict(), plain)
    lightning = tmp_path / "last.ckpt"
    torch.save({"state_dict": model.state_dict(), "epoch": 1}, lightning)
    prefixed = tmp_path / "old.ckpt"
    torch.save(
        {"state_dict": {"model." + k: v for k, v in model.state_dict().items()}},
        prefixed,
    )
    for path in (plain, lightning, prefixed):
        assert list(read_model_state(str(path))) == list(model.state_dict())


def test_legacy_dac_layout_bundle_loads_strictly(tmp_path, caplog):
    """Neither adaptation touches a bundle that needs none (the tiny Encodec)."""
    write_tiny_bundle(tmp_path)
    with caplog.at_level("WARNING"):
        build_codec_from_bundle(read_bundle_artifacts(tmp_path))
    assert "Dropping" not in caplog.text and "Filling" not in caplog.text
