"""Tests for ESPnet3 TTS system stage hooks."""

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

import espnet3.systems.tts.system as sysmod
from espnet3.systems.tts.system import TTSSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# remove_long_short
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_remove_long_short_filters_manifest     | End-to-end duration filter   |
# |                          | keeps in-range rows and drops empty text.       |
# | test_remove_long_short_accepts_single_split_string | splits: "train" is    |
# |                                             | treated as ["train"].        |
# | test_remove_long_short_requires_config      | Missing config sections      |
# |                                             | raise RuntimeError.          |
# | test_remove_long_short_missing_manifest     | Nonexistent manifest raises  |
# |                                             | RuntimeError.                |
# | test_remove_long_short_rejects_stage_args   | Stage arguments raise        |
# |                                             | TypeError.                   |
#
# create_token_list
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_create_token_list_char_tokens          | Char tokens sorted by        |
# |                          | frequency with add_symbol positions honored.    |
# | test_create_token_list_custom_vocab_builder | vocab_builder dotted path    |
# |                                             | fully replaces the default.  |
# | test_create_token_list_requires_config      | Missing config sections      |
# |                                             | raise RuntimeError.          |
# | test_create_token_list_bad_add_symbol       | Malformed add_symbol raises  |
# |                                             | RuntimeError.                |
#
# collect_stats
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_collect_stats_delegates_to_trainer     | Builds the trainer and calls |
# |                                             | trainer.collect_stats().     |
# | test_collect_stats_preserves_null_normalize | model.normalize: null must   |
# |                          | survive into the trainer config (load-bearing). |


def _write_wav(path, seconds, sr=16000):
    frames = int(seconds * sr)
    sf.write(path, np.zeros(frames, dtype=np.float32), sr)


def _write_manifest(tmp_path, name, rows):
    manifest_path = tmp_path / name
    manifest_path.write_text("".join(rows), encoding="utf-8")
    return manifest_path


@pytest.fixture
def duration_manifests(tmp_path):
    """One manifest per split with 0.5s / 2s / 5s wavs and an empty-text row."""
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
            rows.append(f"{utt_id}\t{wav_path}\thello\tspk1\n")
        rows.append(f"{split}_empty\t/no/such.wav\t\tspk1\n")
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
    return TTSSystem(training_config=config)


# ---------------------------------------------------------------
# remove_long_short
# ---------------------------------------------------------------


def test_remove_long_short_filters_manifest(tmp_path, duration_manifests):
    system = _rls_system(tmp_path, duration_manifests)
    system.remove_long_short()

    for split in ("train", "valid"):
        filtered = (tmp_path / "filtered" / f"{split}.tsv").read_text()
        kept_ids = [line.split("\t")[0] for line in filtered.splitlines()]
        # Only the 2s utterance is inside (1.0, 4.0); the empty-text row and
        # the out-of-range wavs are gone.
        assert kept_ids == [f"{split}_mid"]


def test_remove_long_short_accepts_single_split_string(tmp_path, duration_manifests):
    manifests = {"train": duration_manifests["train"]}
    system = _rls_system(tmp_path, manifests, splits="train")
    system.remove_long_short()

    filtered = (tmp_path / "filtered" / "train.tsv").read_text()
    assert [line.split("\t")[0] for line in filtered.splitlines()] == ["train_mid"]


def test_remove_long_short_requires_config(tmp_path):
    system = TTSSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    )
    with pytest.raises(RuntimeError, match="remove_long_short must be set"):
        system.remove_long_short()

    system = TTSSystem(
        training_config=OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "remove_long_short": {}}
        )
    )
    with pytest.raises(RuntimeError, match="save_path must be set"):
        system.remove_long_short()

    system = TTSSystem(
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
# create_token_list
# ---------------------------------------------------------------


def _token_list_system(tmp_path, manifest_path, **overrides):
    tl_cfg = {
        "save_path": str(tmp_path / "tokens"),
        "filename": "tokens.txt",
        "manifest_path": str(manifest_path),
        "token_type": "char",
    }
    tl_cfg.update(overrides)
    config = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "create_token_list": tl_cfg}
    )
    return TTSSystem(training_config=config)


def test_create_token_list_char_tokens(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\taab\tspk1\n", "u2\t/y.wav\tab\tspk1\n"]
    )
    system = _token_list_system(
        tmp_path,
        manifest,
        add_symbol=["<blank>:0", "<unk>:1", "<sos/eos>:-1"],
    )
    system.create_token_list()

    tokens = (tmp_path / "tokens" / "tokens.txt").read_text().splitlines()
    # 'a' occurs 3 times, 'b' twice; special symbols land at 0, 1 and -1.
    assert tokens == ["<blank>", "<unk>", "a", "b", "<sos/eos>"]


def test_create_token_list_custom_vocab_builder(tmp_path):
    manifest = _write_manifest(
        tmp_path, "train.tsv", ["u1\t/x.wav\tbeta\tspk1\n", "u2\t/y.wav\talpha\tspk1\n"]
    )
    # builtins.sorted acts as fn(texts) -> ordered token list, fully
    # replacing the frequency-count construction.
    system = _token_list_system(tmp_path, manifest, vocab_builder="builtins.sorted")
    system.create_token_list()

    tokens = (tmp_path / "tokens" / "tokens.txt").read_text().splitlines()
    assert tokens == ["alpha", "beta"]


def test_create_token_list_requires_config(tmp_path):
    system = TTSSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    )
    with pytest.raises(RuntimeError, match="create_token_list must be set"):
        system.create_token_list()

    system = TTSSystem(
        training_config=OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "create_token_list": {}}
        )
    )
    with pytest.raises(RuntimeError, match="save_path must be set"):
        system.create_token_list()

    system = TTSSystem(
        training_config=OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "create_token_list": {"save_path": str(tmp_path / "tokens")},
            }
        )
    )
    with pytest.raises(RuntimeError, match="filename must be set"):
        system.create_token_list()

    system = _token_list_system(tmp_path, tmp_path / "missing.tsv")
    with pytest.raises(RuntimeError, match="Manifest file not found"):
        system.create_token_list()


def test_create_token_list_bad_add_symbol(tmp_path):
    manifest = _write_manifest(tmp_path, "train.tsv", ["u1\t/x.wav\tab\tspk1\n"])
    system = _token_list_system(tmp_path, manifest, add_symbol=["<blank>"])
    with pytest.raises(RuntimeError, match="Format error"):
        system.create_token_list()


# ---------------------------------------------------------------
# collect_stats
# ---------------------------------------------------------------


class _RecordingTrainer:
    def __init__(self):
        self.collect_stats_calls = 0

    def collect_stats(self):
        self.collect_stats_calls += 1


def _collect_stats_system(tmp_path, monkeypatch):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "seed": 0,
            "model": {"normalize": None, "normalize_conf": None},
        }
    )
    system = TTSSystem(training_config=config)
    trainer = _RecordingTrainer()
    seen_configs = []

    def fake_build_trainer(cfg):
        seen_configs.append(cfg)
        return trainer

    monkeypatch.setattr(sysmod, "_build_trainer", fake_build_trainer)
    monkeypatch.setattr(sysmod, "_ensure_directories", lambda cfg: None)
    return system, trainer, seen_configs


def test_collect_stats_delegates_to_trainer(tmp_path, monkeypatch):
    system, trainer, seen_configs = _collect_stats_system(tmp_path, monkeypatch)
    system.collect_stats()

    assert trainer.collect_stats_calls == 1
    assert seen_configs == [system.training_config]

    with pytest.raises(TypeError):
        system.collect_stats("unexpected")


def test_collect_stats_preserves_null_normalize(tmp_path, monkeypatch):
    """model.normalize: null must reach the trainer config intact.

    The base collect_stats pops normalize/normalize_conf, which resurrects
    espnet2's global_mvn default and breaks stats collection for configs
    that disable normalization on purpose (see TTSSystem.collect_stats
    docstring). The TTS override must not pop them.
    """
    system, _, seen_configs = _collect_stats_system(tmp_path, monkeypatch)
    system.collect_stats()

    model_cfg = seen_configs[0].model
    assert "normalize" in model_cfg
    assert model_cfg.normalize is None
    assert "normalize_conf" in model_cfg
    assert model_cfg.normalize_conf is None
