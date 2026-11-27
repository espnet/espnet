import huggingface_hub
import pytest

from espnet3.systems.snoar.src.net import resolve_model_dir

BUNDLE = [
    "README.md",
    "meta.yaml",
    "conf/inference.yaml",
    "exp/seed/export/config.yaml",
    "exp/seed/export/model.pth",
    "exp/seed/export/snoar.yaml",
]


def _fake_hub(monkeypatch, files, root):
    calls = {}

    def snapshot_download(repo, allow_patterns=None):
        calls["patterns"] = allow_patterns
        return str(root)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo: files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


def test_resolve_model_dir_finds_the_export_of_a_bundle(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, BUNDLE, tmp_path)
    assert resolve_model_dir("espnet/seed") == str(tmp_path / "exp/seed/export")
    assert calls["patterns"] == ["exp/seed/export/*"]


def test_resolve_model_dir_leaves_local_dirs_and_espnet2_tags(monkeypatch, tmp_path):
    assert resolve_model_dir(str(tmp_path)) == str(tmp_path)
    _fake_hub(monkeypatch, ["meta.yaml", "exp/enh_train/valid.acc.pth"], tmp_path)
    assert resolve_model_dir("someone/espnet2_model") == "someone/espnet2_model"


def test_resolve_model_dir_rejects_two_exports(monkeypatch, tmp_path):
    files = BUNDLE + ["other/snoar.yaml", "other/model.pth"]
    _fake_hub(monkeypatch, files, tmp_path)
    with pytest.raises(ValueError, match="several exported models"):
        resolve_model_dir("espnet/seed")
