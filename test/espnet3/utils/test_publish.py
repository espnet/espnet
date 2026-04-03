from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from espnet3.utils import publish


def _make_system(*, exp_dir, publication_config, task=None, inference_config=None):
    training_config = OmegaConf.create({"exp_dir": str(exp_dir), "task": task})
    if inference_config is not None:
        inference_config = OmegaConf.create(inference_config)
    else:
        inference_config = None
    return SimpleNamespace(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )


def test_pack_model_espnet3_manifest(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "dummy.txt").write_text("hello", encoding="utf-8")
    decode_dir = exp_dir / "decode"
    decode_dir.mkdir()
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {"pack_model": {"out_dir": str(out_dir), "decode_dir": str(decode_dir)}}
    )

    system = _make_system(exp_dir=exp_dir, publication_config=publication_config)
    result = publish.pack_model(system)

    assert result == out_dir
    assert out_dir.exists()
    assert (out_dir / "exp" / "dummy.txt").exists()
    meta = OmegaConf.load(out_dir / "meta.yaml")
    assert "files" in meta
    assert "yaml_files" in meta
    assert "timestamp" in meta


def test_pack_model_excludes_decode_and_copies_scores(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "dummy.txt").write_text("hello", encoding="utf-8")
    decode_dir = exp_dir / "decode"
    decode_dir.mkdir()
    (decode_dir / "hyp.scp").write_text("utt1 foo\n", encoding="utf-8")
    scores_path = decode_dir / "scores.json"
    scores_path.write_text('{"wer": 0.1}\n', encoding="utf-8")
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {"pack_model": {"out_dir": str(out_dir), "decode_dir": str(decode_dir)}}
    )

    system = _make_system(exp_dir=exp_dir, publication_config=publication_config)
    result = publish.pack_model(system)

    assert result == out_dir
    assert out_dir.exists()
    assert (out_dir / "exp" / "dummy.txt").exists()
    assert (out_dir / "scores.json").exists()
    assert not (out_dir / "exp" / "decode" / "hyp.scp").exists()


def test_pack_model_espnet2_branch(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    cfg_path = tmp_path / "config.yaml"
    model_path = tmp_path / "model.pth"
    cfg_path.write_text("dummy: true\n", encoding="utf-8")
    model_path.write_text("weights", encoding="utf-8")
    out_dir = tmp_path / "espnet2_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "espnet2": {
                    "task": "asr",
                    "yaml_files": {"asr_train_config": str(cfg_path)},
                    "files": {"asr_model_file": str(model_path)},
                },
            }
        }
    )

    system = _make_system(
        exp_dir=exp_dir, publication_config=publication_config, task="asr"
    )
    result = publish.pack_model(system)

    assert result == out_dir
    assert out_dir.exists()
    meta = OmegaConf.load(out_dir / "meta.yaml")
    assert "files" in meta
    assert "yaml_files" in meta
    assert "timestamp" in meta
