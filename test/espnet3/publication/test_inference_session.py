from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.publication import InferenceSession


class EchoModel:
    def __init__(self, device="cpu", prefix=""):
        self.device = device
        self.prefix = prefix

    def __call__(self, speech):
        return f"{self.prefix}{speech}@{self.device}"


class NoBatchModel:
    def __call__(self, speech):
        if isinstance(speech, list):
            raise TypeError("list input is not supported")
        return f"single:{speech}"


class ImportingModel:
    def __init__(self, asr_train_config, asr_model_file):
        from custom_code import render_value

        self.render_value = render_value
        self.asr_train_config = asr_train_config
        self.asr_model_file = asr_model_file

    def __call__(self, speech):
        return self.render_value(speech)


class DummyDownloader:
    artifacts = None

    def download_and_unpack(self, _model_tag):
        return dict(self.artifacts)


class BackendWithFromPretrained:
    calls = []

    def __init__(self, value):
        self.value = value

    def __call__(self, speech):
        return f"{self.value}:{speech}"

    @staticmethod
    def from_pretrained(model_tag=None, **kwargs):
        BackendWithFromPretrained.calls.append((model_tag, dict(kwargs)))
        return BackendWithFromPretrained(kwargs["value"])


def build_output(*, data, model_output, idx):
    return {"utt_id": data.get("utt_id", idx), "hyp": model_output}


def build_bad_batch_output(*, data, model_output, idx):
    return {"utt_id": "bad", "hyp": model_output}


def _write_bundle_inference_config(
    bundle_root: Path,
    model_target: str,
    *,
    output_fn: str | None = "src.custom_code.build_output",
) -> Path:
    conf_dir = bundle_root / "conf"
    conf_dir.mkdir(exist_ok=True)
    inference_config = conf_dir / "inference.yaml"
    lines = [
        "recipe_dir: .",
        "input_key: speech",
        "model:",
        f"  _target_: {model_target}",
        "  prefix: 'cfg:'",
    ]
    if output_fn is not None:
        lines.append(f"output_fn: {output_fn}")
    lines.append("")
    inference_config.write_text("\n".join(lines), encoding="utf-8")
    return inference_config


def test_from_config_builds_model_and_forward_works():
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "input_key": "speech",
            "model": {
                "_target_": f"{__name__}.EchoModel",
                "prefix": "echo:",
            },
        }
    )

    session = InferenceSession.from_config(cfg)

    assert session("abc") == "echo:abc@cpu"
    assert session({"speech": "xyz"}) == "echo:xyz@cpu"


def test_forward_uses_output_fn_from_config():
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "input_key": "speech",
            "output_fn": f"{__name__}.build_output",
            "model": {
                "_target_": f"{__name__}.EchoModel",
                "prefix": "x:",
            },
        }
    )

    session = InferenceSession.from_config(cfg)
    result = session.forward({"speech": "a", "utt_id": "utt-a"}, idx=7)

    assert result == {"utt_id": "utt-a", "hyp": "x:a@cpu"}


def test_forward_batch_falls_back_to_single_execution():
    session = InferenceSession(
        NoBatchModel(),
        prefer_model_batch=True,
    )

    result = session.forward_batch(["a", "b"])

    assert result == ["single:a", "single:b"]


def test_forward_batch_rejects_length_mismatch():
    session = InferenceSession(EchoModel())

    with pytest.raises(ValueError, match="same length as samples"):
        session.forward_batch(["a"], indices=["x", "y"])


def test_from_pretrained_can_enable_trusted_user_code(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    src_dir = bundle_root / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "custom_code.py").write_text(
        "\n".join(
            [
                "class CustomModel:",
                "    def __init__(self, prefix='', device='cpu'):",
                "        self.prefix = prefix",
                "        self.device = device",
                "",
                "    def __call__(self, speech):",
                "        return f'{self.prefix}{speech}'",
                "",
                "def build_output(*, data, model_output, idx):",
                "    return {'utt_id': data.get('utt_id', idx), 'hyp': model_output}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    inference_config = _write_bundle_inference_config(
        bundle_root,
        "src.custom_code.CustomModel",
    )
    with (bundle_root / "meta.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "files": {},
                "yaml_files": {"inference_config": "conf/inference.yaml"},
                "user_code_paths": ["src"],
            },
            stream,
        )

    DummyDownloader.artifacts = {
        "inference_config": str(inference_config),
    }

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))

    session = InferenceSession.from_pretrained(
        "dummy/tag",
        downloader_class=f"{__name__}.DummyDownloader",
        trust_user_code=True,
    )

    assert session({"speech": "ok", "utt_id": "utt-1"}) == {
        "utt_id": "utt-1",
        "hyp": "cfg:ok",
    }
    assert session.bundle_root == bundle_root.resolve()


def test_from_pretrained_requires_trust_for_user_code(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    src_dir = bundle_root / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "custom_code.py").write_text(
        "class CustomModel:\n"
        "    def __init__(self, prefix='', device='cpu'):\n"
        "        self.prefix = prefix\n"
        "        self.device = device\n"
        "    def __call__(self, speech):\n"
        "        return f'{self.prefix}{speech}'\n",
        encoding="utf-8",
    )
    inference_config = _write_bundle_inference_config(
        bundle_root,
        "src.custom_code.CustomModel",
    )
    with (bundle_root / "meta.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "files": {},
                "yaml_files": {"inference_config": "conf/inference.yaml"},
                "user_code_paths": ["src"],
            },
            stream,
        )
    DummyDownloader.artifacts = {"inference_config": str(inference_config)}

    with pytest.raises(ValueError, match="references bundled user code"):
        InferenceSession.from_pretrained(
            "dummy/tag",
            downloader_class=f"{__name__}.DummyDownloader",
        )


def test_from_pretrained_prefers_backend_from_pretrained_without_user_code():
    BackendWithFromPretrained.calls = []

    session = InferenceSession.from_pretrained(
        "dummy/tag",
        backend_class=f"{__name__}.BackendWithFromPretrained",
        enable_user_code=False,
        value="loaded",
    )

    assert BackendWithFromPretrained.calls == [("dummy/tag", {"value": "loaded"})]
    assert session("x") == "loaded:x"


def test_from_artifacts_keeps_snapshot_bundle_root(tmp_path, monkeypatch):
    model_root = tmp_path / "models--dummy--snapshot-test"
    blobs_dir = model_root / "blobs"
    snapshot_root = model_root / "snapshots" / "rev-1"
    (snapshot_root / "conf").mkdir(parents=True)
    (snapshot_root / "exp" / "train").mkdir(parents=True)
    (snapshot_root / "src").mkdir(parents=True)
    blobs_dir.mkdir(parents=True)

    inference_blob = blobs_dir / "inference-yaml"
    inference_blob.write_text(
        "\n".join(
            [
                "recipe_dir: .",
                "input_key: speech",
                "model:",
                "  _target_: src.custom_code.CustomModel",
                "  asr_train_config: ${recipe_dir}/exp/train/config.yaml",
                "  asr_model_file: ${recipe_dir}/exp/train/last.ckpt",
                "",
            ]
        ),
        encoding="utf-8",
    )
    meta_blob = blobs_dir / "meta-yaml"
    meta_blob.write_text(
        yaml.safe_dump(
            {
                "files": {"asr_model_file": "exp/train/last.ckpt"},
                "yaml_files": {
                    "asr_train_config": "exp/train/config.yaml",
                    "inference_config": "conf/inference.yaml",
                },
                "user_code_paths": ["src"],
            }
        ),
        encoding="utf-8",
    )
    train_config_blob = blobs_dir / "train-config"
    train_config_blob.write_text("dummy: true\n", encoding="utf-8")
    model_blob = blobs_dir / "last-ckpt"
    model_blob.write_text("checkpoint", encoding="utf-8")
    src_init_blob = blobs_dir / "src-init"
    src_init_blob.write_text("", encoding="utf-8")
    src_code_blob = blobs_dir / "src-custom"
    src_code_blob.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "class CustomModel:",
                "    def __init__(self, asr_train_config, asr_model_file, device='cpu'):",
                "        assert Path(asr_train_config).is_file()",
                "        assert Path(asr_model_file).is_file()",
                "        self.device = device",
                "",
                "    def __call__(self, speech):",
                "        return f'snapshot:{speech}@{self.device}'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (snapshot_root / "conf" / "inference.yaml").symlink_to(inference_blob)
    (snapshot_root / "meta.yaml").symlink_to(meta_blob)
    (snapshot_root / "exp" / "train" / "config.yaml").symlink_to(train_config_blob)
    (snapshot_root / "exp" / "train" / "last.ckpt").symlink_to(model_blob)
    (snapshot_root / "src" / "__init__.py").symlink_to(src_init_blob)
    (snapshot_root / "src" / "custom_code.py").symlink_to(src_code_blob)

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))

    session = InferenceSession.from_artifacts(
        {"inference_config": str(snapshot_root / "conf" / "inference.yaml")},
        trust_user_code=True,
    )

    assert session("ok") == "snapshot:ok@cpu"
    assert session.bundle_root == snapshot_root.resolve()


def test_from_artifacts_prefers_inference_config_model_definition(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    inference_config = _write_bundle_inference_config(
        bundle_root,
        f"{__name__}.EchoModel",
        output_fn=None,
    )
    with (bundle_root / "meta.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "files": {},
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            },
            stream,
        )

    session = InferenceSession.from_artifacts(
        {"inference_config": str(inference_config)},
        output_fn_path=None,
        trust_user_code=False,
    )

    assert session("abc") == "cfg:abc@cpu"


def test_forward_batch_rejects_wrong_result_count_from_batch_output_fn():
    session = InferenceSession(
        EchoModel(prefix="b:"),
        output_fn=build_bad_batch_output,
        prefer_model_batch=True,
        fallback_to_single_on_batch_error=False,
    )

    with pytest.raises(RuntimeError, match="wrong number of outputs"):
        session.forward_batch(["a", "b"], use_model_batch=True)
