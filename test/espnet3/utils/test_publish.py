from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.utils import publish


def _make_system(
    *,
    exp_dir,
    publication_config,
    task=None,
    inference_config=None,
    recipe_dir=None,
):
    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "task": task,
            "recipe_dir": str(recipe_dir) if recipe_dir is not None else None,
        }
    )
    if inference_config is not None:
        inference_config = OmegaConf.create(inference_config)
        if recipe_dir is not None:
            inference_config.recipe_dir = str(recipe_dir)
    else:
        inference_config = None
    return SimpleNamespace(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )


@pytest.fixture(autouse=True)
def _stub_publish_runtime(monkeypatch):
    monkeypatch.setattr(publish, "_git_info", lambda: {"head": "", "dirty": ""})
    monkeypatch.setattr(publish, "_hf_username", lambda: "")


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


def test_pack_model_excludes_decode_and_copies_metrics_json_and_readme_table(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "dummy.txt").write_text("hello", encoding="utf-8")
    decode_dir = exp_dir / "decode"
    decode_dir.mkdir()
    (decode_dir / "hyp.scp").write_text("utt1 foo\n", encoding="utf-8")
    metrics_path = decode_dir / "metrics.json"
    metrics_path.write_text(
        (
            "{\n"
            '  "espnet3.systems.asr.metrics.wer.WER": {\n'
            '    "test_clean": {"WER": 12.3},\n'
            '    "test_other": {"WER": 23.4}\n'
            "  },\n"
            '  "espnet3.systems.asr.metrics.cer.CER": {\n'
            '    "test_clean": {"CER": 4.5},\n'
            '    "test_other": {"CER": 8.9}\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "decode_dir": str(decode_dir),
                "readme_context": {
                    "task": "asr",
                    "lang": "en",
                    "license": "apache-2.0",
                    "description": "Example description.",
                },
            }
        }
    )

    system = _make_system(exp_dir=exp_dir, publication_config=publication_config)
    result = publish.pack_model(system)

    assert result == out_dir
    assert out_dir.exists()
    assert (out_dir / "exp" / "dummy.txt").exists()
    assert (out_dir / "metrics.json").exists()
    assert not (out_dir / "exp" / "decode" / "hyp.scp").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "- asr" in readme_text
    assert "| Test | CER | WER |" in readme_text
    assert "| test_clean | 4.5 | 12.3 |" in readme_text
    assert "| test_other | 8.9 | 23.4 |" in readme_text


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


def test_pack_model_espnet3_includes_recipe_assets_and_relative_manifest(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "train_debug"
    conf_dir = recipe_dir / "conf"
    src_dir = recipe_dir / "src"
    exp_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)

    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (exp_dir / "last.ckpt").write_text("weights\n", encoding="utf-8")
    (conf_dir / "training.yaml").write_text(
        "recipe_dir: .\nexp_tag: train_debug\nexp_dir: ${recipe_dir}/exp/${exp_tag}\n",
        encoding="utf-8",
    )
    (conf_dir / "inference.yaml").write_text(
        "recipe_dir: .\n"
        "input_key: speech\n"
        "model:\n"
        "  _target_: espnet2.bin.asr_inference.Speech2Text\n"
        "  asr_train_config: ${recipe_dir}/exp/train_debug/config.yaml\n"
        "  asr_model_file: ${recipe_dir}/exp/train_debug/last.ckpt\n"
        "output_fn: src.inference.build_output\n",
        encoding="utf-8",
    )
    (src_dir / "inference.py").write_text(
        "def build_output(*, data, model_output, idx):\n"
        "    return {'utt_id': data.get('utt_id', idx), 'hyp': model_output}\n",
        encoding="utf-8",
    )
    (recipe_dir / "run.py").write_text("print('run')\n", encoding="utf-8")
    (recipe_dir / "pixi.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create({"pack_model": {"out_dir": str(out_dir)}})
    system = _make_system(
        exp_dir=exp_dir,
        publication_config=publication_config,
        recipe_dir=recipe_dir,
        inference_config={"input_key": "speech"},
    )

    result = publish.pack_model(system)

    assert result == out_dir
    assert (out_dir / "conf" / "inference.yaml").exists()
    assert (out_dir / "src" / "inference.py").exists()
    assert (out_dir / "run.py").exists()
    assert (out_dir / "pixi.toml").exists()
    meta = OmegaConf.to_container(OmegaConf.load(out_dir / "meta.yaml"), resolve=True)
    assert meta["yaml_files"]["inference_config"] == "conf/inference.yaml"
    assert meta["yaml_files"]["training_config"] == "conf/training.yaml"
    assert meta["user_code_paths"] == ["src"]


def test_pack_model_prunes_unreferenced_yaml_files(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "train_debug"
    conf_dir = recipe_dir / "conf"
    tuning_dir = conf_dir / "tuning"
    exp_dir.mkdir(parents=True)
    tuning_dir.mkdir(parents=True)

    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (exp_dir / "unused.yaml").write_text("unused: true\n", encoding="utf-8")
    (conf_dir / "settings.yaml").write_text("exp_tag: train_debug\n", encoding="utf-8")
    (conf_dir / "training.yaml").write_text(
        "recipe_dir: .\nexp_tag: train_debug\nexp_dir: ${recipe_dir}/exp/${exp_tag}\n",
        encoding="utf-8",
    )
    (tuning_dir / "unused.yaml").write_text("foo: bar\n", encoding="utf-8")

    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create({"pack_model": {"out_dir": str(out_dir)}})
    system = _make_system(
        exp_dir=exp_dir,
        publication_config=publication_config,
        recipe_dir=recipe_dir,
        inference_config={
            "exp_tag": "${load_yaml:settings.yaml,exp_tag}",
            "input_key": "speech",
            "model": {
                "_target_": "espnet2.bin.asr_inference.Speech2Text",
                "asr_train_config": str(exp_dir / "config.yaml"),
            },
        },
    )

    result = publish.pack_model(system)

    assert result == out_dir
    assert (out_dir / "conf" / "settings.yaml").exists()
    assert (out_dir / "conf" / "inference.yaml").exists()
    assert (out_dir / "exp" / "train_debug" / "config.yaml").exists()
    assert not (out_dir / "conf" / "tuning" / "unused.yaml").exists()
    assert not (out_dir / "exp" / "train_debug" / "unused.yaml").exists()


def test_pack_model_espnet2_defaults_include_runtime_artifacts(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "train_debug"
    conf_dir = recipe_dir / "conf"
    src_dir = recipe_dir / "src"
    data_dir = recipe_dir / "data" / "bpe_30"
    exp_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (exp_dir / "config.yaml").write_text(
        "recipe_dir: .\nexp_dir: ./exp/train_debug\ndata_dir: ./data\n",
        encoding="utf-8",
    )
    (exp_dir / "last.ckpt").write_text("weights\n", encoding="utf-8")
    (conf_dir / "training.yaml").write_text("exp_tag: train_debug\n", encoding="utf-8")
    (conf_dir / "inference.yaml").write_text(
        "exp_tag: ${load_yaml:training.yaml,exp_tag}\n"
        "model:\n"
        "  _target_: espnet2.bin.asr_inference.Speech2Text\n"
        "  asr_train_config: ${exp_dir}/config.yaml\n"
        "  asr_model_file: ${exp_dir}/last.ckpt\n"
        "output_fn: src.inference.build_output\n",
        encoding="utf-8",
    )
    (src_dir / "inference.py").write_text(
        "def build_output(*, data, model_output, idx):\n"
        "    return {'hyp': model_output}\n",
        encoding="utf-8",
    )
    (data_dir / "tokens.txt").write_text("<blank>\na\n", encoding="utf-8")
    (data_dir / "bpe.model").write_text("model\n", encoding="utf-8")

    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "data_dir": str(recipe_dir / "data"),
            "exp_tag": "train_debug",
            "recipe_dir": str(recipe_dir),
            "task": "espnet3.systems.asr.task.ASRTask",
            "tokenizer": {"save_path": str(data_dir)},
            "token_list": "./data/bpe_30/tokens.txt",
            "bpemodel": "./data/bpe_30/bpe.model",
        }
    )
    inference_config = OmegaConf.create(
        {
            "recipe_dir": str(recipe_dir),
            "exp_dir": str(exp_dir),
            "inference_dir": str(exp_dir / "inference"),
            "model": {
                "_target_": "espnet2.bin.asr_inference.Speech2Text",
                "asr_train_config": str(exp_dir / "config.yaml"),
                "asr_model_file": str(exp_dir / "last.ckpt"),
            },
            "output_fn": "src.inference.build_output",
        }
    )
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "model_pack"),
                "espnet2": {"task": "asr"},
            }
        }
    )
    system = SimpleNamespace(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )

    out_dir = publish.pack_model(system)

    assert (out_dir / "exp" / "train_debug" / "config.yaml").exists()
    assert (out_dir / "exp" / "train_debug" / "last.ckpt").exists()
    assert (out_dir / "data" / "bpe_30" / "tokens.txt").exists()
    meta = OmegaConf.to_container(OmegaConf.load(out_dir / "meta.yaml"), resolve=True)
    assert meta["yaml_files"]["asr_train_config"] == "exp/train_debug/config.yaml"
    assert meta["files"]["asr_model_file"] == "exp/train_debug/last.ckpt"
    bundle_inference = (out_dir / "conf" / "inference.yaml").read_text(encoding="utf-8")
    assert "${recipe_dir}/exp/train_debug/config.yaml" in bundle_inference
    assert "${recipe_dir}/exp/train_debug/last.ckpt" in bundle_inference


def test_pack_model_can_skip_data_dir_and_keep_tokenizer_files(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "train_debug"
    conf_dir = recipe_dir / "conf"
    tokenizer_dir = recipe_dir / "data" / "bpe_30"
    manifest_dir = recipe_dir / "data" / "manifest"
    exp_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    (exp_dir / "dummy.txt").write_text("hello\n", encoding="utf-8")
    (conf_dir / "training.yaml").write_text("exp_tag: train_debug\n", encoding="utf-8")
    (tokenizer_dir / "tokens.txt").write_text("<blank>\na\n", encoding="utf-8")
    (tokenizer_dir / "bpe.model").write_text("model\n", encoding="utf-8")
    (tokenizer_dir / "train_tokenizer.log").write_text("log\n", encoding="utf-8")
    (manifest_dir / "train.tsv").write_text("utt\ttext\n", encoding="utf-8")

    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "data_dir": str(recipe_dir / "data"),
            "recipe_dir": str(recipe_dir),
        }
    )
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "model_pack"),
                "include_data_dir": False,
                "extra": [str(tokenizer_dir)],
                "exclude": ["**/*.log"],
            }
        }
    )
    system = SimpleNamespace(
        training_config=training_config,
        inference_config=None,
        publication_config=publication_config,
    )

    out_dir = publish.pack_model(system)

    assert (out_dir / "data" / "bpe_30" / "tokens.txt").exists()
    assert (out_dir / "data" / "bpe_30" / "bpe.model").exists()
    assert not (out_dir / "data" / "bpe_30" / "train_tokenizer.log").exists()
    assert not (out_dir / "data" / "manifest" / "train.tsv").exists()


def test_pack_model_preserves_symlink_name_in_bundle_config(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "train_debug"
    conf_dir = recipe_dir / "conf"
    src_dir = recipe_dir / "src"
    exp_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)

    (exp_dir / "config.yaml").write_text(
        "recipe_dir: .\nexp_dir: ./exp/train_debug\ndata_dir: ./data\n",
        encoding="utf-8",
    )
    (exp_dir / "step1.ckpt").write_text("weights\n", encoding="utf-8")
    (exp_dir / "last.ckpt").symlink_to("step1.ckpt")
    (conf_dir / "training.yaml").write_text("exp_tag: train_debug\n", encoding="utf-8")
    (conf_dir / "inference.yaml").write_text(
        "exp_tag: ${load_yaml:training.yaml,exp_tag}\n"
        "model:\n"
        "  _target_: espnet2.bin.asr_inference.Speech2Text\n"
        "  asr_train_config: ${exp_dir}/config.yaml\n"
        "  asr_model_file: ${exp_dir}/last.ckpt\n"
        "output_fn: src.inference.build_output\n",
        encoding="utf-8",
    )
    (src_dir / "inference.py").write_text(
        "def build_output(*, data, model_output, idx):\n"
        "    return {'hyp': model_output}\n",
        encoding="utf-8",
    )

    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "data_dir": str(recipe_dir / "data"),
            "exp_tag": "train_debug",
            "recipe_dir": str(recipe_dir),
            "task": "espnet3.systems.asr.task.ASRTask",
        }
    )
    inference_config = OmegaConf.create(
        {
            "recipe_dir": str(recipe_dir),
            "exp_dir": str(exp_dir),
            "model": {
                "_target_": "espnet2.bin.asr_inference.Speech2Text",
                "asr_train_config": str(exp_dir / "config.yaml"),
                "asr_model_file": str(exp_dir / "last.ckpt"),
            },
            "output_fn": "src.inference.build_output",
        }
    )
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "model_pack"),
                "espnet2": {"task": "asr"},
            }
        }
    )
    system = SimpleNamespace(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )

    out_dir = publish.pack_model(system)

    meta = OmegaConf.to_container(OmegaConf.load(out_dir / "meta.yaml"), resolve=True)
    assert meta["files"]["asr_model_file"] == "exp/train_debug/last.ckpt"
    bundle_inference = (out_dir / "conf" / "inference.yaml").read_text(encoding="utf-8")
    assert "${recipe_dir}/exp/train_debug/last.ckpt" in bundle_inference
    assert "step1.ckpt" not in bundle_inference
