from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.publication.demo import packing as demo_packer
from espnet3.systems.asr.system import ASRSystem
from espnet3.utils import publish
from espnet3.utils.publish import (
    _build_results_table,
    _render_readme,
    _resolve_results,
)


def _make_system(
    *,
    exp_dir,
    recipe_dir,
    publication_config,
    task=None,
    inference_config=None,
):
    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "task": task,
            "recipe_dir": str(recipe_dir),
        }
    )
    if inference_config is not None:
        inference_config = OmegaConf.create(inference_config)
        inference_config.recipe_dir = str(recipe_dir)
    return SimpleNamespace(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )


# ---------------------------------------------------------------------------
# _resolve_results
# ---------------------------------------------------------------------------


def test_resolve_results_returns_none_when_all_configs_none():
    assert _resolve_results(None, None, None) is None


def test_resolve_results_finds_metrics_in_publication_config(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    cfg = OmegaConf.create({"inference_dir": str(tmp_path)})

    result = _resolve_results(cfg, None, None)

    assert result == metrics_path


def test_resolve_results_finds_metrics_in_metrics_config(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    cfg = OmegaConf.create({"inference_dir": str(tmp_path)})

    result = _resolve_results(None, cfg, None)

    assert result == metrics_path


def test_resolve_results_finds_metrics_in_inference_config(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    cfg = OmegaConf.create({"inference_dir": str(tmp_path)})

    result = _resolve_results(None, None, cfg)

    assert result == metrics_path


def test_resolve_results_returns_none_when_inference_dir_has_no_metrics(tmp_path):
    cfg = OmegaConf.create({"inference_dir": str(tmp_path)})

    assert _resolve_results(cfg, None, None) is None


def test_resolve_results_returns_none_when_no_inference_dir():
    cfg = OmegaConf.create({"other_key": "value"})

    assert _resolve_results(cfg, None, None) is None


def test_resolve_results_prioritises_publication_over_metrics(tmp_path):
    pub_dir = tmp_path / "pub"
    met_dir = tmp_path / "met"
    pub_dir.mkdir()
    met_dir.mkdir()
    pub_metrics = pub_dir / "metrics.json"
    met_metrics = met_dir / "metrics.json"
    pub_metrics.write_text("{}", encoding="utf-8")
    met_metrics.write_text("{}", encoding="utf-8")
    pub_cfg = OmegaConf.create({"inference_dir": str(pub_dir)})
    met_cfg = OmegaConf.create({"inference_dir": str(met_dir)})

    result = _resolve_results(pub_cfg, met_cfg, None)

    assert result == pub_metrics


# ---------------------------------------------------------------------------
# _build_results_table
# ---------------------------------------------------------------------------


def test_build_results_table_returns_empty_for_none():
    assert _build_results_table(None) == ""


def test_build_results_table_returns_empty_for_nonexistent_file(tmp_path):
    assert _build_results_table(tmp_path / "missing.json") == ""


def test_build_results_table_returns_empty_for_malformed_json(tmp_path):
    bad = tmp_path / "metrics.json"
    bad.write_text("not json", encoding="utf-8")
    assert _build_results_table(bad) == ""


def test_build_results_table_returns_empty_for_empty_results(tmp_path):
    empty = tmp_path / "metrics.json"
    empty.write_text("{}", encoding="utf-8")
    assert _build_results_table(empty) == ""


def test_build_results_table_returns_empty_when_all_values_are_non_dict(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"metric": "not_a_dict"}), encoding="utf-8")
    assert _build_results_table(metrics_file) == ""


def test_build_results_table_with_nested_dict_values(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        json.dumps({"WER": {"test_clean": {"WER": 5.0}}}), encoding="utf-8"
    )

    table = _build_results_table(metrics_file)

    assert "| dataset | WER |" in table
    assert "| test_clean | 5.0 |" in table


def test_build_results_table_with_flat_scalar_values(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        json.dumps({"my.module.WER": {"test_clean": 5.0}}), encoding="utf-8"
    )

    table = _build_results_table(metrics_file)

    assert "WER" in table
    assert "test_clean" in table
    assert "5.0" in table


def test_build_results_table_with_multiple_metrics_and_test_sets(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "mod.WER": {
                    "test_clean": {"WER": 5.0},
                    "test_other": {"WER": 10.0},
                },
                "mod.CER": {
                    "test_clean": {"CER": 2.0},
                    "test_other": {"CER": 4.0},
                },
            }
        ),
        encoding="utf-8",
    )

    table = _build_results_table(metrics_file)

    assert "| dataset | CER | WER |" in table
    assert "| test_clean | 2.0 | 5.0 |" in table
    assert "| test_other | 4.0 | 10.0 |" in table


def test_build_results_table_uses_short_metric_name(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        json.dumps({"espnet3.systems.asr.metrics.wer.WER": {"test": 3.0}}),
        encoding="utf-8",
    )

    table = _build_results_table(metrics_file)

    assert "WER" in table
    assert "espnet3" not in table


def test_build_results_table_rows_are_sorted(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "WER": {
                    "b_test": {"WER": 2.0},
                    "a_test": {"WER": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )

    table = _build_results_table(metrics_file)
    lines = [
        line
        for line in table.splitlines()
        if line.startswith("| ") and "dataset" not in line and "---" not in line
    ]

    assert lines[0].startswith("| a_test")
    assert lines[1].startswith("| b_test")


# ---------------------------------------------------------------------------
# _render_readme
# ---------------------------------------------------------------------------


def test_render_readme_substitutes_placeholders():
    template = "Hello ${name}!"
    result = _render_readme(template, {"name": "World"})
    assert result == "Hello World!"


def test_render_readme_drops_line_with_empty_placeholder():
    template = "keep\n${missing_key} drop this\nkeep too"
    result = _render_readme(template, {})
    assert "drop this" not in result
    assert "keep" in result
    assert "keep too" in result


def test_render_readme_keeps_line_with_filled_placeholder():
    template = "tag: ${task}"
    result = _render_readme(template, {"task": "asr"})
    assert result == "tag: asr"


def test_render_readme_drops_line_when_one_of_multiple_placeholders_empty():
    template = "${a} and ${b}"
    result = _render_readme(template, {"a": "x", "b": ""})
    assert result.strip() == ""


def test_render_readme_keeps_line_with_no_placeholders():
    template = "no placeholders here"
    result = _render_readme(template, {})
    assert result == "no placeholders here"


def test_render_readme_treats_none_value_as_empty_and_drops_line():
    template = "line: ${key}"
    result = _render_readme(template, {"key": None})
    assert "line:" not in result


# ---------------------------------------------------------------------------
# pack_model – basic integration
# ---------------------------------------------------------------------------


def test_pack_model_creates_output_dir_and_copies_exp_dir(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    (exp_dir / "dummy.txt").write_text("hello", encoding="utf-8")
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "readme": "egs3/TEMPLATE/asr/src/hf_model_repo_readme_template.md",
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    result = publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
        inference_config=system.inference_config,
    )

    assert result == out_dir
    assert (out_dir / "exp" / "dummy.txt").exists()
    meta = OmegaConf.load(out_dir / "meta.yaml")
    assert "files" in meta
    assert "yaml_files" in meta
    assert "torch" in meta


def test_pack_model_resolves_repo_root_readme_path_outside_cwd(tmp_path, monkeypatch):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "readme": "egs3/TEMPLATE/asr/src/hf_model_repo_readme_template.md",
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )
    monkeypatch.chdir(tmp_path)

    publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    assert (out_dir / "README.md").exists()


def test_pack_model_recreates_existing_out_dir(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    out_dir = tmp_path / "model_pack"
    out_dir.mkdir()
    stale_file = out_dir / "stale.txt"
    stale_file.write_text("old", encoding="utf-8")

    publication_config = OmegaConf.create({"pack_model": {"out_dir": str(out_dir)}})
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    assert not stale_file.exists()


def test_pack_model_raises_when_artifact_file_does_not_exist(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "files": {"asr_model_file": str(exp_dir / "last.ckpt")},
                "yaml_files": {"asr_train_config": str(exp_dir / "config.yaml")},
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    with pytest.raises(RuntimeError, match="Artifact does not exist"):
        publish.pack_model(
            training_config=system.training_config,
            publication_config=OmegaConf.create(
                {
                    "pack_model": {
                        "out_dir": str(tmp_path / "pack"),
                        "files": {"model": str(recipe_dir / "missing.ckpt")},
                    }
                }
            ),
        )


def test_pack_model_raises_when_artifact_outside_recipe_dir(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp"
    recipe_dir.mkdir()
    exp_dir.mkdir()
    outside_file = tmp_path / "outside.ckpt"
    outside_file.write_text("weights", encoding="utf-8")
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "files": {"asr_model_file": str(exp_dir / "last.ckpt")},
                "yaml_files": {"asr_train_config": str(exp_dir / "config.yaml")},
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    with pytest.raises(RuntimeError, match="recipe_dir"):
        publish.pack_model(
            training_config=system.training_config,
            publication_config=OmegaConf.create(
                {
                    "pack_model": {
                        "out_dir": str(tmp_path / "pack"),
                        "files": {"model": str(outside_file)},
                    }
                }
            ),
        )


def test_pack_model_raises_when_readme_template_missing(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "readme": "nonexistent/template.md",
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    with pytest.raises(FileNotFoundError, match="README template not found"):
        publish.pack_model(
            training_config=system.training_config,
            publication_config=system.publication_config,
        )


def test_pack_model_excludes_inference_dir_and_copies_metrics_json(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    inference_dir = exp_dir / "inference"
    inference_dir.mkdir()
    (inference_dir / "hyp.scp").write_text("utt1 foo\n", encoding="utf-8")
    (inference_dir / "metrics.json").write_text(
        json.dumps({"WER": {"test": {"WER": 5.0}}}), encoding="utf-8"
    )
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "upload_model": {"hf_repo": "espnet/test-repo"},
            "pack_model": {
                "out_dir": str(out_dir),
                "exclude": ["inference"],
                "readme": "egs3/TEMPLATE/asr/src/hf_model_repo_readme_template.md",
                "readme_context": {
                    "task": "asr",
                    "lang": "en",
                    "license": "apache-2.0",
                    "description": "Example.",
                },
            },
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    result = publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    assert (result / "metrics.json").exists()
    assert not (result / "exp" / "inference" / "hyp.scp").exists()
    readme = (result / "README.md").read_text(encoding="utf-8")
    assert "# ESPnet3 asr model" in readme
    assert "| dataset | WER |" in readme
    assert "| test | 5.0 |" in readme
    assert "Metrics were not bundled." not in readme
    assert 'from_pretrained("espnet/test-repo", trust_user_code=True)' in readme


def test_pack_model_excludes_recursive_log_and_tensorboard_patterns(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    src_dir = recipe_dir / "src"
    exp_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (exp_dir / "keep.txt").write_text("keep\n", encoding="utf-8")
    (exp_dir / "train.log").write_text("drop\n", encoding="utf-8")
    (exp_dir / "nested").mkdir()
    (exp_dir / "nested" / "collect.log").write_text("drop\n", encoding="utf-8")
    (exp_dir / "tensorboard" / "tb_logger").mkdir(parents=True)
    (exp_dir / "tensorboard" / "tb_logger" / "events.out").write_text(
        "drop\n", encoding="utf-8"
    )
    (src_dir / "__pycache__").mkdir()
    (src_dir / "__pycache__" / "mod.pyc").write_text("drop\n", encoding="utf-8")
    (src_dir / "inference.py").write_text("keep\n", encoding="utf-8")

    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "include": [str(src_dir)],
                "exclude": ["**/*.log", "**/tensorboard/**"],
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    out_dir = publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    assert (out_dir / "exp" / "run" / "keep.txt").exists()
    assert not (out_dir / "exp" / "run" / "train.log").exists()
    assert not (out_dir / "exp" / "run" / "nested" / "collect.log").exists()
    assert not (out_dir / "exp" / "run" / "tensorboard").exists()
    assert not (out_dir / "src" / "__pycache__").exists()
    assert (out_dir / "src" / "inference.py").exists()


def test_pack_model_excludes_inference_dirs_without_dropping_src_module(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    src_dir = recipe_dir / "src"
    inference_dir = exp_dir / "inference"
    inference_transducer_dir = exp_dir / "inference_transducer"
    exp_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    inference_dir.mkdir()
    inference_transducer_dir.mkdir()
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "inference.py").write_text("keep\n", encoding="utf-8")
    (inference_dir / "hyp.scp").write_text("utt1 foo\n", encoding="utf-8")
    (inference_transducer_dir / "hyp.scp").write_text("utt1 bar\n", encoding="utf-8")

    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "include": [str(src_dir)],
                "exclude": ["inference", "inference_*"],
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    out_dir = publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    assert not (out_dir / "exp" / "run" / "inference").exists()
    assert not (out_dir / "exp" / "run" / "inference_transducer").exists()
    assert (out_dir / "src" / "inference.py").exists()


def test_pack_model_writes_readme_with_full_context(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "readme": "egs3/TEMPLATE/asr/src/hf_model_repo_readme_template.md",
                "readme_context": {
                    "task": "asr",
                    "lang": "en",
                    "license": "apache-2.0",
                    "description": "My model.",
                },
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )

    publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "# ESPnet3 asr model" in readme
    assert "- Repository:" not in readme
    assert "language: en" in readme
    assert "license: apache-2.0" in readme
    assert "My model." in readme
    assert (
        'model = InferenceModel.from_packed("/path/to/packed_model", '
        "trust_user_code=True)" in readme
    )


def test_pack_model_drops_readme_lines_for_missing_context(
    tmp_path, caplog, monkeypatch
):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    out_dir = tmp_path / "model_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "readme": "egs3/TEMPLATE/asr/src/hf_model_repo_readme_template.md",
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir,
        recipe_dir=recipe_dir,
        publication_config=publication_config,
        task="espnet3.systems.asr.task.ASRTask",
    )
    monkeypatch.setattr(
        publish,
        "get_git_metadata",
        lambda cwd=None: {"short_commit": "abc123", "worktree": "dirty"},
    )
    monkeypatch.setenv("USER", "tester")

    publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "# ESPnet3 asr model" in readme
    assert "- System: `espnet3.systems.asr.task.ASRTask`" in readme
    assert "- Creator: `tester`" in readme
    assert "- Git: `abc123` (dirty)" in readme
    assert "language:" not in readme
    assert "license:" not in readme
    assert "Metrics were not bundled." in readme
    assert "Run the `measure` stage before `pack_model`" in readme
    assert "README will not include a rendered results table" in caplog.text


def test_pack_model_with_explicit_artifacts(tmp_path):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    exp_dir.mkdir()
    cfg_path = recipe_dir / "config.yaml"
    model_path = recipe_dir / "model.pth"
    cfg_path.write_text("dummy: true\n", encoding="utf-8")
    model_path.write_text("weights", encoding="utf-8")
    out_dir = tmp_path / "espnet2_pack"
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(out_dir),
                "files": {"asr_model_file": str(model_path)},
                "yaml_files": {"asr_train_config": str(cfg_path)},
            }
        }
    )
    system = _make_system(
        exp_dir=exp_dir,
        recipe_dir=recipe_dir,
        publication_config=publication_config,
        task="asr",
    )

    result = publish.pack_model(
        training_config=system.training_config,
        publication_config=system.publication_config,
    )

    meta = OmegaConf.to_container(OmegaConf.load(result / "meta.yaml"), resolve=True)
    assert meta["files"]["asr_model_file"] == "model.pth"
    assert meta["yaml_files"]["asr_train_config"] == "config.yaml"


def test_pack_model_writes_bundle_config_with_relative_paths(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    conf_dir = recipe_dir / "conf"
    exp_dir.mkdir(parents=True)
    conf_dir.mkdir()
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (exp_dir / "last.ckpt").write_text("weights\n", encoding="utf-8")
    (conf_dir / "inference.yaml").write_text(
        "recipe_dir: .\n"
        "input_key: speech\n"
        "model:\n"
        "  _target_: espnet2.bin.asr_inference.Speech2Text\n"
        "  asr_train_config: ${recipe_dir}/exp/run/config.yaml\n"
        "  asr_model_file: ${recipe_dir}/exp/run/last.ckpt\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "pack"
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "recipe_dir": str(recipe_dir)}
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
        }
    )
    publication_config = OmegaConf.create({"pack_model": {"out_dir": str(out_dir)}})
    system = ASRSystem(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )

    out_dir = system.pack_model()

    bundle_inference = (out_dir / "conf" / "inference.yaml").read_text(encoding="utf-8")
    assert "${recipe_dir}/exp/run/config.yaml" in bundle_inference
    assert "${recipe_dir}/exp/run/last.ckpt" in bundle_inference


def test_pack_model_includes_extra_data_dir(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    data_dir = recipe_dir / "data" / "bpe_30"
    exp_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (exp_dir / "last.ckpt").write_text("weights\n", encoding="utf-8")
    (data_dir / "tokens.txt").write_text("<blank>\na\n", encoding="utf-8")

    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
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
        }
    )
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "include": [str(recipe_dir / "data")],
            }
        }
    )
    system = ASRSystem(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )

    out_dir = system.pack_model()

    assert (out_dir / "data" / "bpe_30" / "tokens.txt").exists()


def test_pack_model_expands_globbed_include_paths(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    data_dir = recipe_dir / "data" / "bpe_30"
    manifest_dir = recipe_dir / "data" / "manifest"
    exp_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")
    (exp_dir / "last.ckpt").write_text("weights\n", encoding="utf-8")
    (data_dir / "bpe.model").write_text("spm\n", encoding="utf-8")
    (data_dir / "tokens.txt").write_text("<blank>\na\n", encoding="utf-8")
    (manifest_dir / "train.tsv").write_text("utt\tpath\n", encoding="utf-8")

    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
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
        }
    )
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "include": [
                    str(recipe_dir / "data" / "**" / "bpe.model"),
                    str(recipe_dir / "data" / "**" / "tokens.txt"),
                ],
            }
        }
    )
    system = ASRSystem(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )

    out_dir = system.pack_model()

    assert (out_dir / "data" / "bpe_30" / "bpe.model").exists()
    assert (out_dir / "data" / "bpe_30" / "tokens.txt").exists()
    assert not (out_dir / "data" / "manifest").exists()


def test_pack_model_preserves_symlink_name(tmp_path):
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    exp_dir.mkdir(parents=True)
    (exp_dir / "step1.ckpt").write_text("weights\n", encoding="utf-8")
    (exp_dir / "last.ckpt").symlink_to("step1.ckpt")
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")

    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
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
        }
    )
    publication_config = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": str(tmp_path / "pack"),
                "files": {"asr_model_file": str(exp_dir / "last.ckpt")},
                "yaml_files": {"asr_train_config": str(exp_dir / "config.yaml")},
            }
        }
    )
    system = ASRSystem(
        training_config=training_config,
        inference_config=inference_config,
        publication_config=publication_config,
    )

    out_dir = system.pack_model()

    meta = OmegaConf.to_container(OmegaConf.load(out_dir / "meta.yaml"), resolve=True)
    assert meta["files"]["asr_model_file"] == "exp/run/last.ckpt"
    bundle = (out_dir / "conf" / "inference.yaml").read_text(encoding="utf-8")
    assert "${recipe_dir}/exp/run/last.ckpt" in bundle
    assert "step1.ckpt" not in bundle


def test_upload_common_creates_missing_repo_with_options_and_uploads(
    tmp_path, monkeypatch
):
    src_dir = tmp_path / "demo"
    src_dir.mkdir()
    calls = []

    monkeypatch.setattr(publish.shutil, "which", lambda cmd: "/usr/bin/huggingface-cli")
    monkeypatch.setattr(
        publish, "_check_repo_exists", lambda repo, repo_type="model": False
    )
    monkeypatch.setattr(publish, "_run", lambda cmd, cwd=None: calls.append(cmd) or "")

    publish._upload_common(
        "espnet/test-demo",
        src_dir,
        repo_type="space",
        create_options={"organization": "espnet", "space_sdk": "gradio"},
        create_repo_name="test-demo",
    )

    assert calls == [
        [
            "huggingface-cli",
            "repo",
            "create",
            "test-demo",
            "--type",
            "space",
            "-y",
            "--organization",
            "espnet",
            "--space_sdk",
            "gradio",
        ],
        [
            "huggingface-cli",
            "upload",
            "espnet/test-demo",
            str(src_dir),
            "--repo-type",
            "space",
        ],
    ]


def test_upload_model_uses_shared_upload_helper(tmp_path, monkeypatch):
    recipe_dir = tmp_path
    exp_dir = recipe_dir / "exp"
    pack_dir = tmp_path / "model_pack"
    exp_dir.mkdir()
    pack_dir.mkdir()
    publication_config = OmegaConf.create(
        {
            "upload_model": {"hf_repo": "espnet/test-repo"},
            "pack_model": {"out_dir": str(pack_dir)},
        }
    )
    system = _make_system(
        exp_dir=exp_dir, recipe_dir=recipe_dir, publication_config=publication_config
    )
    calls = []

    monkeypatch.setattr(
        publish,
        "_upload_common",
        lambda repo, src_dir, **kwargs: calls.append((repo, src_dir, kwargs)),
    )

    publish.upload_model(system)

    assert calls == [("espnet/test-repo", pack_dir, {"repo_type": "model"})]


def test_upload_demo_uses_shared_upload_helper(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    demo_config = OmegaConf.create(
        {
            "pack": {"out_dir": str(demo_dir)},
            "upload_demo": {
                "hf_repo": "espnet/test-demo",
                "organization": "espnet",
            },
        }
    )
    system = SimpleNamespace(demo_config=demo_config, exp_dir=None)
    calls = []

    monkeypatch.setattr(
        publish,
        "_upload_common",
        lambda repo, src_dir, **kwargs: calls.append((repo, src_dir, kwargs)),
    )

    demo_packer.upload_demo(system)

    assert calls == [
        (
            "espnet/test-demo",
            demo_dir,
            {
                "repo_type": "space",
                "create_options": {
                    "organization": "espnet",
                    "space_sdk": "gradio",
                    "yes": True,
                },
                "create_repo_name": "test-demo",
            },
        )
    ]
