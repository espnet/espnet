"""Tests for ESPnet3 CLS system stage hooks."""

import logging

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

import espnet3.systems.cls.system as sysmod
from espnet3.systems.cls.system import CLSSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# remove_long_short
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_remove_long_short_filters_manifest     | End-to-end duration filter   |
# |                          | keeps in-range rows and drops empty labels.     |
# | test_remove_long_short_accepts_single_split_string | splits: "train" is    |
# |                                             | treated as ["train"].        |
# | test_remove_long_short_default_manifest_location | Without manifest_paths  |
# |                       | the stage reads data/manifest/{split}.tsv.         |
# | test_remove_long_short_rereads_durations_on_rerun | New bounds must not   |
# |                          | reuse shard results written under the old ones. |
# | test_remove_long_short_sets_parallel        | A parallel config section is |
# |                                             | forwarded to set_parallel.   |
# | test_remove_long_short_requires_config      | Missing config sections      |
# |                                             | raise RuntimeError.          |
# | test_remove_long_short_missing_manifest     | Nonexistent manifest raises  |
# |                                             | RuntimeError.                |
# | test_remove_long_short_rejects_stage_args   | Stage arguments raise        |
# |                                             | TypeError.                   |
#
# prepare_labels
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_prepare_labels_orders_by_frequency     | Descending count order, as   |
# |                                             | in ESPnet2's cls.sh stage 4. |
# | test_prepare_labels_keeps_labels_verbatim   | Labels reach the token list  |
# |                          | exactly as the manifest spells them, so they    |
# |                          | still match the reference labels.               |
# | test_prepare_labels_splits_multi_label_rows | Whitespace in the label      |
# |                          | column contributes every label it carries.      |
# | test_prepare_labels_add_symbol_positions    | Negative indices count from  |
# |                                             | the end.                     |
# | test_prepare_labels_warns_about_rows_without_a_label | Rows carrying no    |
# |                          | label are counted and reported, blank lines     |
# |                          | are not.                                        |
# | test_prepare_labels_default_manifest_location | Without manifest_path the  |
# |                       | stage reads data/manifest/train.tsv.               |
# | test_prepare_labels_requires_config         | Missing config sections      |
# |                                             | raise RuntimeError.          |
# | test_prepare_labels_missing_manifest        | Nonexistent manifest raises  |
# |                                             | RuntimeError.                |
# | test_prepare_labels_empty_manifest          | A manifest with no label     |
# |                                             | raises RuntimeError.         |
# | test_prepare_labels_bad_add_symbol          | Malformed add_symbol raises  |
# |                                             | RuntimeError.                |
# | test_prepare_labels_rejects_stage_args      | Stage arguments raise        |
# |                                             | TypeError.                   |
#
# Stage log directories
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_stage_log_dirs_cover_both_stages       | Both added stages log into   |
# |                                             | their own save_path.         |
# | test_stage_log_dirs_honour_caller_overrides | A caller-supplied mapping    |
# |                                             | wins over the defaults.      |


def _write_wav(path, seconds, sr=16000):
    frames = int(seconds * sr)
    sf.write(path, np.zeros(frames, dtype=np.float32), sr)


def _write_manifest(tmp_path, name, rows):
    manifest_path = tmp_path / name
    manifest_path.write_text("".join(rows), encoding="utf-8")
    return manifest_path


@pytest.fixture
def duration_manifests(tmp_path):
    """One manifest per split with 0.5s / 2s / 5s wavs and an empty-label row."""
    manifests = {}
    for split in ("train", "valid"):
        rows = []
        for utt_id, seconds in (
            (f"{split}_short", 0.5),
            (f"{split}_mid", 2.0),
            (f"{split}_long", 5.0),
        ):
            wav_path = tmp_path / f"{utt_id}.wav"
            _write_wav(wav_path, seconds)
            rows.append(f"{utt_id}\t{wav_path}\thappy\n")
        rows.append(f"{split}_empty\t/no/such.wav\t\n")
        manifests[split] = str(_write_manifest(tmp_path, f"{split}.tsv", rows))
    return manifests


def _rls_system(tmp_path, manifests, **overrides):
    rls = {
        "save_path": str(tmp_path / "filtered"),
        "min_wav_duration": 1.0,
        "max_wav_duration": 4.0,
        "splits": list(manifests.keys()),
        "manifest_paths": manifests,
    }
    rls.update(overrides)
    config = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "remove_long_short": rls}
    )
    return CLSSystem(training_config=config)


def _labels_system(tmp_path, manifest_path=None, **overrides):
    cfg = {
        "save_path": str(tmp_path / "data"),
        "filename": "token_list",
    }
    if manifest_path is not None:
        cfg["manifest_path"] = str(manifest_path)
    cfg.update(overrides)
    config = OmegaConf.create({"exp_dir": str(tmp_path / "exp"), "prepare_labels": cfg})
    return CLSSystem(training_config=config)


def _token_list(tmp_path):
    return (tmp_path / "data" / "token_list").read_text().splitlines()


# ---------------------------------------------------------------
# remove_long_short
# ---------------------------------------------------------------


def test_remove_long_short_filters_manifest(tmp_path, duration_manifests):
    system = _rls_system(tmp_path, duration_manifests)
    system.remove_long_short()

    for split in ("train", "valid"):
        filtered = (tmp_path / "filtered" / f"{split}.tsv").read_text()
        kept_ids = [line.split("\t")[0] for line in filtered.splitlines()]
        # Only the 2s utterance is inside (1.0, 4.0); the empty-label row and
        # the out-of-range wavs are gone.
        assert kept_ids == [f"{split}_mid"]


def test_remove_long_short_accepts_single_split_string(tmp_path, duration_manifests):
    manifests = {"train": duration_manifests["train"]}
    system = _rls_system(tmp_path, manifests, splits="train")
    system.remove_long_short()

    filtered = (tmp_path / "filtered" / "train.tsv").read_text()
    assert [line.split("\t")[0] for line in filtered.splitlines()] == ["train_mid"]


def test_remove_long_short_default_manifest_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "data" / "manifest"
    manifest_dir.mkdir(parents=True)
    wav_path = tmp_path / "mid.wav"
    _write_wav(wav_path, 2.0)
    manifest_dir.joinpath("train.tsv").write_text(
        f"utt_mid\t{wav_path}\thappy\n", encoding="utf-8"
    )

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": {
                "save_path": str(tmp_path / "filtered"),
                "min_wav_duration": 1.0,
                "max_wav_duration": 4.0,
                "splits": ["train"],
            },
        }
    )
    CLSSystem(training_config=config).remove_long_short()

    filtered = (tmp_path / "filtered" / "train.tsv").read_text()
    assert [line.split("\t")[0] for line in filtered.splitlines()] == ["utt_mid"]


def test_remove_long_short_rereads_durations_on_rerun(tmp_path, duration_manifests):
    """Widening the bounds must not reuse the previous run's keep/drop shards."""
    manifests = {"train": duration_manifests["train"]}
    _rls_system(tmp_path, manifests, splits=["train"]).remove_long_short()

    _rls_system(
        tmp_path, manifests, splits=["train"], max_wav_duration=10.0
    ).remove_long_short()

    filtered = (tmp_path / "filtered" / "train.tsv").read_text()
    assert [line.split("\t")[0] for line in filtered.splitlines()] == [
        "train_mid",
        "train_long",
    ]


def test_remove_long_short_sets_parallel(tmp_path, duration_manifests, monkeypatch):
    calls = []
    monkeypatch.setattr(sysmod, "set_parallel", lambda cfg: calls.append(cfg))

    manifests = {"train": duration_manifests["train"]}
    system = _rls_system(tmp_path, manifests, splits=["train"])
    system.training_config.parallel = OmegaConf.create({"env": "local"})
    system.remove_long_short()

    assert calls == [system.training_config.parallel]


def test_remove_long_short_requires_config(tmp_path):
    system = CLSSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    )
    with pytest.raises(RuntimeError, match="remove_long_short must be set"):
        system.remove_long_short()

    system = CLSSystem(
        training_config=OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "remove_long_short": {}}
        )
    )
    with pytest.raises(RuntimeError, match="save_path must be set"):
        system.remove_long_short()

    system = CLSSystem(
        training_config=OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "remove_long_short": {"save_path": str(tmp_path / "filtered")},
            }
        )
    )
    with pytest.raises(RuntimeError, match="min_wav_duration"):
        system.remove_long_short()


def test_remove_long_short_missing_manifest(tmp_path):
    manifests = {"train": str(tmp_path / "missing.tsv")}
    system = _rls_system(tmp_path, manifests)
    with pytest.raises(RuntimeError, match="Manifest file not found"):
        system.remove_long_short()


def test_remove_long_short_rejects_stage_args(tmp_path, duration_manifests):
    system = _rls_system(tmp_path, duration_manifests)
    with pytest.raises(TypeError):
        system.remove_long_short("unexpected")


# ---------------------------------------------------------------
# prepare_labels
# ---------------------------------------------------------------


def test_prepare_labels_orders_by_frequency(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        [
            "u1\t/x.wav\tneutral\n",
            "u2\t/y.wav\thappy\n",
            "u3\t/z.wav\tneutral\n",
            "u4\t/w.wav\tneutral\n",
            "u5\t/v.wav\thappy\n",
            "u6\t/u.wav\tsad\n",
        ],
    )
    _labels_system(tmp_path, manifest).prepare_labels()

    assert _token_list(tmp_path) == ["neutral", "happy", "sad"]


def test_prepare_labels_keeps_labels_verbatim(tmp_path):
    """Ensure labels reach the token list exactly as the manifest spells them.

    The reference labels used for scoring are read from the same column, so
    a label rewritten only here would no longer match them.
    """
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        ["u1\t/x.wav\tNot_OK\n", "u2\t/y.wav\tsurprise!\n"],
    )
    _labels_system(tmp_path, manifest).prepare_labels()

    assert sorted(_token_list(tmp_path)) == ["Not_OK", "surprise!"]


def test_prepare_labels_splits_multi_label_rows(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        ["u1\t/x.wav\thappy surprise\n", "u2\t/y.wav\thappy\n"],
    )
    _labels_system(tmp_path, manifest).prepare_labels()

    assert _token_list(tmp_path) == ["happy", "surprise"]


def test_prepare_labels_add_symbol_positions(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\thappy\n", "u2\t/y.wav\tsad\n"]
    )
    _labels_system(
        tmp_path, manifest, add_symbol=["<blank>:0", "<unk>:-1"]
    ).prepare_labels()

    # -1 appends past the end, matching ESPnet2's convention.
    assert _token_list(tmp_path) == ["<blank>", "happy", "sad", "<unk>"]


def test_prepare_labels_warns_about_rows_without_a_label(tmp_path, caplog):
    """Ensure rows carrying no label are counted and reported.

    They are only left out of the token list; the manifest still feeds them
    to training. A blank line is not a row, so it is not counted.
    """
    manifest = _write_manifest(
        tmp_path,
        "train.tsv",
        ["u1\t/x.wav\n", "u2\t/y.wav\thappy\n", "u3\t/z.wav\t\n", "\n"],
    )

    with caplog.at_level(logging.WARNING):
        _labels_system(tmp_path, manifest).prepare_labels()

    assert _token_list(tmp_path) == ["happy"]
    assert "2 row(s)" in caplog.text


def test_prepare_labels_default_manifest_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "data" / "manifest"
    manifest_dir.mkdir(parents=True)
    manifest_dir.joinpath("train.tsv").write_text(
        "u1\t/x.wav\thappy\n", encoding="utf-8"
    )

    _labels_system(tmp_path).prepare_labels()

    assert _token_list(tmp_path) == ["happy"]


def test_prepare_labels_requires_config(tmp_path):
    system = CLSSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    )
    with pytest.raises(RuntimeError, match="prepare_labels must be set"):
        system.prepare_labels()

    system = CLSSystem(
        training_config=OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "prepare_labels": {}}
        )
    )
    with pytest.raises(RuntimeError, match="save_path must be set"):
        system.prepare_labels()

    system = CLSSystem(
        training_config=OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "prepare_labels": {"save_path": str(tmp_path / "data")},
            }
        )
    )
    with pytest.raises(RuntimeError, match="filename must be set"):
        system.prepare_labels()


def test_prepare_labels_missing_manifest(tmp_path):
    system = _labels_system(tmp_path, tmp_path / "missing.tsv")
    with pytest.raises(RuntimeError, match="Manifest file not found"):
        system.prepare_labels()


def test_prepare_labels_empty_manifest(tmp_path):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\t\n"])
    system = _labels_system(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="No label found"):
        system.prepare_labels()


def test_prepare_labels_bad_add_symbol(tmp_path):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\thappy\n"])
    system = _labels_system(tmp_path, manifest, add_symbol=["<unk>"])
    with pytest.raises(RuntimeError, match="Format error"):
        system.prepare_labels()


def test_prepare_labels_rejects_stage_args(tmp_path):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\thappy\n"])
    system = _labels_system(tmp_path, manifest)
    with pytest.raises(TypeError):
        system.prepare_labels("unexpected")


# ---------------------------------------------------------------
# Stage log directories
# ---------------------------------------------------------------


def test_stage_log_dirs_cover_both_stages(tmp_path):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": {"save_path": str(tmp_path / "filtered")},
            "prepare_labels": {"save_path": str(tmp_path / "data")},
        }
    )
    log_dirs = CLSSystem(training_config=config).stage_log_dirs

    assert log_dirs["remove_long_short"] == tmp_path / "filtered"
    assert log_dirs["prepare_labels"] == tmp_path / "data"


def test_stage_log_dirs_honour_caller_overrides(tmp_path):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "prepare_labels": {"save_path": str(tmp_path / "data")},
        }
    )
    log_dirs = CLSSystem(
        training_config=config,
        stage_log_mapping={"prepare_labels": "training_config.exp_dir"},
    ).stage_log_dirs

    assert log_dirs["prepare_labels"] == tmp_path / "exp"
