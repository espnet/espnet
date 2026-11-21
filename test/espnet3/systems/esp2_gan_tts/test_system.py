"""Tests for espnet3.systems.esp2_gan_tts.system (ESP2GANTTSSystem)."""

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.esp2_gan_tts.system as sysmod
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.contract.stages import stage_log_dir
from espnet3.systems.esp2_gan_tts.system import ESP2GANTTSSystem
from espnet3.systems.f5tts.system import F5TTSSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# _build_trainer dispatch and the train / collect_stats stages
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_trainer_dispatches_gan_models    | AbsGANESPnetModel goes to    |
# |                                             | build_gan_trainer.           |
# | test_build_trainer_uses_plain_trainer       | Non-GAN models get the plain |
# |                                             | ESPnet3LightningTrainer.     |
# | test_build_trainer_instantiates_model_once  | The model is built exactly   |
# |                          | once, so the global RNG is not advanced twice.   |
# | test_train_builds_trainer_and_fits          | train() forwards the fit     |
# |                                             | kwargs to trainer.fit().     |
# | test_train_without_fit_section              | A missing/empty fit section  |
# |                                             | calls fit() with no kwargs.  |
# | test_train_saves_espnet_config_for_task     | save_espnet_config runs only |
# |                                             | when training_config.task is set. |
# | test_train_rejects_stage_args               | Stage arguments raise        |
# |                                             | TypeError.                   |
# | test_collect_stats_uses_gan_aware_trainer   | collect_stats() builds the   |
# |                          | trainer through this module's _build_trainer.    |
# | test_collect_stats_keeps_normalize_in_model | The normalize keys reach the |
# |                                             | trainer config untouched.    |
# | test_prepare_training_runtime               | Directories, parallel, seed  |
# |                                             | and matmul precision are set.|
#
# Stage declaration
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_declares_the_stage_order               | compute_xvectors runs right  |
# |                          | after create_dataset; the rest is F5TTSSystem's. |
# | test_stage_log_dirs_follow_the_training_config | Each data stage logs under|
# |                          | its save_path, compute_xvectors under xvector's. |
#
# compute_xvectors
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_compute_xvectors_runs_each_split       | One provider/runner pair per |
# |                          | split, wired with the manifest and output dirs.  |
# | test_compute_xvectors_accepts_single_split  | splits: "train" is treated   |
# |                                             | as ["train"].                |
# | test_compute_xvectors_default_manifest      | Without manifest_paths the   |
# |                          | stage reads data/manifest/{split}.tsv.           |
# | test_compute_xvectors_null_manifest_paths   | An explicit null mapping     |
# |                                             | falls back the same way.     |
# | test_compute_xvectors_requires_config       | Missing xvector/save_path    |
# |                                             | raise RuntimeError.          |
# | test_compute_xvectors_missing_manifest      | A nonexistent manifest       |
# |                                             | raises RuntimeError.         |
# | test_compute_xvectors_rejects_empty_manifest | An empty manifest raises.   |
# | test_compute_xvectors_sets_parallel         | A parallel config section is |
# |                                             | forwarded to set_parallel.   |
# | test_compute_xvectors_rejects_stage_args    | Stage arguments raise.       |
# | test_compute_xvectors_counts_merged_results | The merged shard records are |
# |                                             | counted by status.           |


# ---------------------------------------------------------------
# _build_trainer dispatch and the train / collect_stats stages
# ---------------------------------------------------------------


class _FakeGANModel(AbsGANESPnetModel):
    def forward(self, forward_generator: bool = True, **batch):
        raise NotImplementedError

    def collect_feats(self, **batch):
        raise NotImplementedError


def _trainer_config(tmp_path):
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "best_model_criterion": [["valid/loss", 3, "min"]],
            "trainer": {"accelerator": "cpu"},
            "model": {"_target_": "dummy.Model"},
        }
    )


def test_build_trainer_dispatches_gan_models(tmp_path, monkeypatch):
    """A GAN model is wrapped by the GAN Lightning module and trainer."""
    config = _trainer_config(tmp_path)
    model = _FakeGANModel()
    monkeypatch.setattr(sysmod, "_instantiate_model", lambda cfg: model)

    seen = []
    monkeypatch.setattr(
        sysmod,
        "build_gan_trainer",
        lambda cfg, m: seen.append((cfg, m)) or "gan-trainer",
    )
    monkeypatch.setattr(
        sysmod,
        "ESPnet3LightningTrainer",
        lambda **kwargs: pytest.fail("plain trainer must not be used for GAN models"),
    )

    assert sysmod._build_trainer(config) == "gan-trainer"
    assert seen == [(config, model)]


def test_build_trainer_uses_plain_trainer(tmp_path, monkeypatch):
    """A plain model gets the plain Lightning module and trainer."""
    config = _trainer_config(tmp_path)
    model = object()
    monkeypatch.setattr(sysmod, "_instantiate_model", lambda cfg: model)
    monkeypatch.setattr(sysmod, "ESPnetLightningModule", lambda m, cfg: ("lit", m, cfg))

    seen = []
    monkeypatch.setattr(
        sysmod,
        "ESPnet3LightningTrainer",
        lambda **kwargs: seen.append(kwargs) or "plain-trainer",
    )

    assert sysmod._build_trainer(config) == "plain-trainer"
    assert seen == [
        {
            "model": ("lit", model, config),
            "exp_dir": config.exp_dir,
            "config": config.trainer,
            "best_model_criterion": config.best_model_criterion,
        }
    ]


def test_build_trainer_instantiates_model_once(tmp_path, monkeypatch):
    """Instantiating a model twice would advance the RNG and change training.

    The GAN dispatch is deliberately not implemented by delegating to the base
    builder, precisely so the model is created exactly once.
    """
    config = _trainer_config(tmp_path)
    calls = []

    def fake_instantiate(cfg):
        calls.append(cfg)
        return _FakeGANModel()

    monkeypatch.setattr(sysmod, "_instantiate_model", fake_instantiate)
    monkeypatch.setattr(sysmod, "build_gan_trainer", lambda cfg, m: "gan-trainer")

    sysmod._build_trainer(config)

    assert calls == [config]


class _RecordingTrainer:
    def __init__(self):
        self.fit_calls = []
        self.collect_stats_calls = 0

    def fit(self, **kwargs):
        self.fit_calls.append(kwargs)

    def collect_stats(self):
        self.collect_stats_calls += 1


def _train_system(tmp_path, monkeypatch, extra=None):
    config = _trainer_config(tmp_path)
    config.seed = 0
    if extra:
        config.merge_with(OmegaConf.create(extra))

    system = ESP2GANTTSSystem(training_config=config)
    trainer = _RecordingTrainer()
    seen_configs = []

    def fake_build_trainer(cfg):
        seen_configs.append(cfg)
        return trainer

    monkeypatch.setattr(sysmod, "_build_trainer", fake_build_trainer)
    monkeypatch.setattr(
        ESP2GANTTSSystem, "_prepare_training_runtime", lambda self: None, raising=True
    )
    return system, trainer, seen_configs


def test_train_builds_trainer_and_fits(tmp_path, monkeypatch):
    """The train stage builds the trainer once and forwards the ``fit`` section."""
    system, trainer, seen_configs = _train_system(
        tmp_path, monkeypatch, extra={"fit": {"ckpt_path": "last"}}
    )

    system.train()

    assert seen_configs == [system.training_config]
    assert trainer.fit_calls == [{"ckpt_path": "last"}]


def test_train_without_fit_section(tmp_path, monkeypatch):
    """A missing or empty ``fit`` section means ``fit()`` with no arguments."""
    system, trainer, _ = _train_system(tmp_path, monkeypatch)
    system.train()
    assert trainer.fit_calls == [{}]

    system, trainer, _ = _train_system(tmp_path, monkeypatch, extra={"fit": {}})
    system.train()
    assert trainer.fit_calls == [{}]


def test_train_saves_espnet_config_for_task(tmp_path, monkeypatch):
    """The train stage writes the espnet2-style config when ``task`` is set."""
    saved = []
    monkeypatch.setattr(
        sysmod,
        "save_espnet_config",
        lambda task, cfg, exp_dir: saved.append((task, exp_dir)),
    )

    system, _, _ = _train_system(tmp_path, monkeypatch)
    system.train()
    assert saved == []

    system, _, _ = _train_system(tmp_path, monkeypatch, extra={"task": "tts"})
    system.train()
    assert saved == [("tts", system.training_config.exp_dir)]


def test_train_rejects_stage_args(tmp_path, monkeypatch):
    """Stages take no positional or keyword arguments."""
    system, _, _ = _train_system(tmp_path, monkeypatch)

    with pytest.raises(TypeError):
        system.train("unexpected")
    with pytest.raises(TypeError):
        system.train(unexpected=True)


def test_collect_stats_uses_gan_aware_trainer(tmp_path, monkeypatch):
    """The stats stage must not fall back to the base (non-GAN) builder."""
    system, trainer, seen_configs = _train_system(tmp_path, monkeypatch)

    system.collect_stats()

    assert seen_configs == [system.training_config]
    assert trainer.collect_stats_calls == 1
    with pytest.raises(TypeError):
        system.collect_stats("unexpected")


def test_collect_stats_keeps_normalize_in_model(tmp_path, monkeypatch):
    """`normalize: null` must reach the task builder; popping it restores GlobalMVN."""
    system, _, seen_configs = _train_system(
        tmp_path,
        monkeypatch,
        extra={"model": {"normalize": None, "normalize_conf": {}}},
    )

    system.collect_stats()

    model_config = seen_configs[0].model
    assert "normalize" in model_config and model_config.normalize is None
    assert "normalize_conf" in model_config


def test_prepare_training_runtime(tmp_path, monkeypatch):
    """The runtime setup seeds, sets parallelism and creates the directories."""
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "seed": 777,
            "parallel": {"env": "local"},
        }
    )
    calls = {}
    monkeypatch.setattr(
        sysmod, "set_parallel", lambda arg: calls.setdefault("parallel", arg)
    )
    monkeypatch.setattr(
        sysmod.L,
        "seed_everything",
        lambda seed, workers=True: calls.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        sysmod.torch,
        "set_float32_matmul_precision",
        lambda p: calls.setdefault("precision", p),
    )

    ESP2GANTTSSystem(training_config=config)._prepare_training_runtime()

    assert (tmp_path / "exp").is_dir() and (tmp_path / "stats").is_dir()
    assert calls == {"parallel": config.parallel, "seed": 777, "precision": "high"}

    # No parallel / seed configured, stats_dir left null: only the exp dir.
    calls.clear()
    bare = OmegaConf.create({"exp_dir": str(tmp_path / "exp2"), "stats_dir": None})
    ESP2GANTTSSystem(training_config=bare)._prepare_training_runtime()
    assert (tmp_path / "exp2").is_dir()
    assert calls == {"precision": "high"}


# ---------------------------------------------------------------
# Stage declaration
# ---------------------------------------------------------------


def test_declares_the_stage_order():
    """compute_xvectors runs right after create_dataset; the rest is F5TTSSystem's."""
    assert issubclass(ESP2GANTTSSystem, F5TTSSystem)
    assert [spec.name for spec in ESP2GANTTSSystem.stages] == [
        "create_dataset",
        "compute_xvectors",
        "remove_long_short",
        "create_token_list",
        "collect_stats",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
        "pack_demo",
        "upload_demo",
    ]
    xvector = ESP2GANTTSSystem.stages[1]
    assert (xvector.config, xvector.log_dir) == ("training", "xvector.save_path")
    assert ESP2GANTTSSystem.stages[2:] == F5TTSSystem.stages[1:]


def test_stage_log_dirs_follow_the_training_config(tmp_path):
    """Each custom stage's log_dir points at its save_path in the training config."""
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "remove_long_short": {"save_path": str(tmp_path / "filtered")},
            "create_token_list": {"save_path": str(tmp_path / "tokens")},
            "xvector": {"save_path": str(tmp_path / "xvectors")},
        }
    )
    system = ESP2GANTTSSystem(training_config=config)

    assert stage_log_dir(system, "compute_xvectors") == tmp_path / "xvectors"
    assert stage_log_dir(system, "remove_long_short") == tmp_path / "filtered"
    assert stage_log_dir(system, "create_token_list") == tmp_path / "tokens"
    assert stage_log_dir(system, "train") == tmp_path / "exp"

    # Without an xvector block the stage falls back to the default log dir.
    plain = ESP2GANTTSSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp2")})
    )
    assert stage_log_dir(plain, "compute_xvectors") == Path(tmp_path / "exp2")


# ---------------------------------------------------------------
# compute_xvectors
# ---------------------------------------------------------------


class _RecordingRunner:
    """Stand-in for XVectorRunner that records how it was driven."""

    calls = []
    results = None

    def __init__(
        self,
        provider=None,
        batch_size=None,
        output_dir=None,
        shard_subdir="",
        resume=True,
    ):
        self.provider = provider
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.shard_subdir = shard_subdir
        self.resume = resume

    def __call__(self, indices):
        type(self).calls.append(
            {
                "indices": list(indices),
                "batch_size": self.batch_size,
                "output_dir": str(self.output_dir),
                "shard_subdir": self.shard_subdir,
                "resume": self.resume,
                "config": self.provider.config,
                "params": dict(self.provider.params),
            }
        )
        if type(self).results is None:
            return [{"idx": i, "utt_id": f"u{i}", "status": "ok"} for i in indices]
        return type(self).results


def _xvector_manifest(tmp_path, name, n=2):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"u{i}\t/x{i}.wav\thello\t0\n" for i in range(n)), encoding="utf-8"
    )
    return path


def _xvector_system(tmp_path, monkeypatch, xvector=None, manifests=("train",)):
    _RecordingRunner.calls = []
    _RecordingRunner.results = None
    monkeypatch.setattr(sysmod, "XVectorRunner", _RecordingRunner)

    xvector_config = {
        "save_path": str(tmp_path / "x_vectors"),
        "splits": list(manifests),
        "manifest_paths": {
            split: str(_xvector_manifest(tmp_path, f"{split}.tsv"))
            for split in manifests
        },
    }
    if xvector is not None:
        xvector_config.update(xvector)
    config = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "xvector": xvector_config}
    )
    return ESP2GANTTSSystem(training_config=config)


def test_compute_xvectors_runs_each_split(tmp_path, monkeypatch):
    """The x-vector stage runs the runner once per configured split."""
    system = _xvector_system(
        tmp_path,
        monkeypatch,
        xvector={
            "toolkit": "speechbrain",
            "pretrained_model": "speechbrain/spkrec-ecapa-voxceleb",
            "device": "cpu",
            "spk_embed_tag": "ecapa",
            "batch_size": 8,
        },
        manifests=("train", "valid"),
    )

    system.compute_xvectors()

    assert len(_RecordingRunner.calls) == 2
    first = _RecordingRunner.calls[0]
    assert first["indices"] == [0, 1]
    assert first["batch_size"] == 8
    # The provider reads toolkit/model/device from the training config itself;
    # the stage only tells it which manifest to read and where to write.
    assert set(first["params"]) == {"manifest_path", "xvector_dir"}
    assert first["params"]["manifest_path"].endswith("train.tsv")
    assert first["config"] is system.training_config
    # Embeddings land in one directory per split, tagged by the model; the
    # runner's shard bookkeeping goes beside them, under save_path/shards.
    assert first["params"]["xvector_dir"].endswith("ecapa_train")
    assert _RecordingRunner.calls[1]["params"]["xvector_dir"].endswith("ecapa_valid")
    assert first["output_dir"] == str(tmp_path / "x_vectors" / "shards")
    assert first["shard_subdir"] == "ecapa_train"
    assert first["resume"] is False


def test_compute_xvectors_accepts_single_split(tmp_path, monkeypatch):
    """A single split name is accepted in place of a list."""
    system = _xvector_system(tmp_path, monkeypatch)
    system.training_config.xvector.splits = "train"

    system.compute_xvectors()

    assert len(_RecordingRunner.calls) == 1


def test_compute_xvectors_default_manifest(tmp_path, monkeypatch):
    """Without manifest_paths the stage reads data/manifest/<split>.tsv."""
    monkeypatch.chdir(tmp_path)
    _xvector_manifest(tmp_path, "data/manifest/train.tsv")
    _RecordingRunner.calls = []
    _RecordingRunner.results = None
    monkeypatch.setattr(sysmod, "XVectorRunner", _RecordingRunner)

    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "xvector": {
                "save_path": str(tmp_path / "x_vectors"),
                "splits": ["train"],
            },
        }
    )
    ESP2GANTTSSystem(training_config=config).compute_xvectors()

    manifest_used = _RecordingRunner.calls[0]["params"]["manifest_path"]
    assert manifest_used.endswith("data/manifest/train.tsv")


def test_compute_xvectors_null_manifest_paths(tmp_path, monkeypatch):
    """``manifest_paths:`` left empty in YAML is None, not a missing key."""
    monkeypatch.chdir(tmp_path)
    _xvector_manifest(tmp_path, "data/manifest/train.tsv")
    system = _xvector_system(tmp_path, monkeypatch)
    system.training_config.xvector.manifest_paths = None

    system.compute_xvectors()

    manifest_used = _RecordingRunner.calls[0]["params"]["manifest_path"]
    assert manifest_used.endswith("data/manifest/train.tsv")


def test_compute_xvectors_requires_config(tmp_path, monkeypatch):
    """A training config without an ``xvector`` section is a RuntimeError."""
    monkeypatch.setattr(sysmod, "XVectorRunner", _RecordingRunner)

    no_xvector = ESP2GANTTSSystem(
        training_config=OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    )
    with pytest.raises(RuntimeError, match="training_config.xvector must be set"):
        no_xvector.compute_xvectors()

    no_save_path = ESP2GANTTSSystem(
        training_config=OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "xvector": {"splits": ["train"]}}
        )
    )
    with pytest.raises(RuntimeError, match="xvector.save_path must be set"):
        no_save_path.compute_xvectors()


def test_compute_xvectors_missing_manifest(tmp_path, monkeypatch):
    """A manifest path that does not exist is a RuntimeError."""
    system = _xvector_system(tmp_path, monkeypatch)
    system.training_config.xvector.manifest_paths.train = str(tmp_path / "nope.tsv")

    with pytest.raises(RuntimeError, match="Manifest file not found"):
        system.compute_xvectors()


def test_compute_xvectors_rejects_empty_manifest(tmp_path, monkeypatch):
    """An empty manifest is a RuntimeError, not a silent no-op."""
    system = _xvector_system(tmp_path, monkeypatch)
    empty = tmp_path / "empty.tsv"
    empty.write_text("", encoding="utf-8")
    system.training_config.xvector.manifest_paths.train = str(empty)

    with pytest.raises(RuntimeError, match="No utterances found in manifest"):
        system.compute_xvectors()


def test_compute_xvectors_sets_parallel(tmp_path, monkeypatch):
    """The stage applies the training config's ``parallel`` section."""
    system = _xvector_system(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(sysmod, "set_parallel", lambda cfg: calls.append(cfg))
    system.training_config.parallel = OmegaConf.create({"env": "local"})

    system.compute_xvectors()

    assert calls == [system.training_config.parallel]


def test_compute_xvectors_rejects_stage_args(tmp_path, monkeypatch):
    """The x-vector stage takes no arguments either."""
    system = _xvector_system(tmp_path, monkeypatch)

    with pytest.raises(TypeError):
        system.compute_xvectors("unexpected")


def test_compute_xvectors_counts_merged_results(tmp_path, monkeypatch, caplog):
    """The merged shard records are counted by status for the log."""
    system = _xvector_system(tmp_path, monkeypatch)
    _RecordingRunner.results = [
        {"idx": 0, "utt_id": "u0", "status": "ok"},
        {"idx": 1, "utt_id": "u1", "status": "skipped"},
    ]

    with caplog.at_level(logging.INFO):
        system.compute_xvectors()

    assert len(_RecordingRunner.calls) == 1
    assert "(1 new, 1 skipped)" in caplog.text
