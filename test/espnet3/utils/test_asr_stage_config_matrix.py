"""Exercise ASR stage configuration resolution and persistence end to end."""

import logging
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch
from espnet3.utils.run_utils import ConfigError
from espnet3.utils.stage_configs import resolve_stage_configs

_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "test_utils"
    / "espnet3"
    / "stage_config_matrix"
)
_CASES = OmegaConf.to_container(OmegaConf.load(_FIXTURES / "cases.yaml"))
_STAGES = list(_CASES)


class ASRConfigCase:
    """Materialize input configs and saved histories in an isolated experiment."""

    def __init__(self, root):
        self.root = root
        self.configs = {
            p.stem: OmegaConf.create(p.read_text().replace("__ROOT__", str(root)))
            for p in (_FIXTURES / "conf").glob("*.yaml")
        }
        self.snapshots = OmegaConf.create(
            (_FIXTURES / "snapshots.yaml").read_text().replace("__ROOT__", str(root))
        )
        self.exp = root / "train"
        self.exp.mkdir()
        self.calls = []
        self.observed = {}

    def load_snapshot(self, stage):
        config = OmegaConf.create(OmegaConf.to_container(self.snapshots[stage]))
        OmegaConf.resolve(config)
        return config

    def get_stage_directory(self, stage):
        return Path(
            OmegaConf.select(self.load_snapshot(stage), _CASES[stage]["directory_key"])
        )

    def get_directory_overrides(self):
        return {
            "exp_dir": str(self.exp),
            "data_dir": str(self.root / "data"),
            "stats_dir": str(self.root / "stats"),
            "tokenizer.save_path": str(self.root / "tokenizer"),
            "inference_dir": str(self.exp / "inference"),
            "pack.out_dir": str(self.root / "demo"),
        }

    def save_history(self, stages):
        for stage in stages:
            config = self.load_snapshot(stage)
            directory = self.get_stage_directory(stage)
            directory.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, directory / f"{stage}.yaml")
            spec = next(s for s in ASRSystem.stages if s.name == stage)
            roles = (spec.config,) if isinstance(spec.config, str) else spec.config
            own_keys = {key for role in roles for key in self.configs[role]}
            own = OmegaConf.create({key: config[key] for key in own_keys})
            baked = self.exp / "config" / f"{stage}.yaml"
            baked.parent.mkdir(exist_ok=True)
            OmegaConf.save(own, baked)

    def execute(self, stages, overrides=None, configs=None):
        args = ["--stages", *stages]
        selected = self.configs if configs is None else configs
        for role, config in selected.items():
            path = self.root / f"{role}.yaml"
            OmegaConf.save(config, path)
            args += [f"--{role}_config", str(path)]
        for key, value in (overrides or {}).items():
            flag = (
                "--exp_dir"
                if key == "exp_dir"
                else "--" + key.replace(".", "-").replace("_", "-")
            )
            args += [flag, str(value)]
        launch(ASRSystem, argv=args)


@pytest.fixture(name="case")
def create_case(tmp_path, monkeypatch):
    case = ASRConfigCase(tmp_path)
    # Keep real launch, config resolution, stage execution and persistence. Only
    # replace template loading and the expensive/external stage operations.
    monkeypatch.setattr(
        "espnet3.systems.launch.load_and_merge_config",
        lambda path, **kwargs: OmegaConf.load(path),
    )
    monkeypatch.setattr(
        "espnet3.systems.launch.configure_logging",
        lambda: logging.getLogger("test.stage_matrix"),
    )
    for stage in _STAGES:

        def record(system, stage=stage):
            case.calls.append(stage)
            case.observed[stage] = OmegaConf.to_container(
                system.stage_configs[stage], resolve=True
            )

        monkeypatch.setattr(ASRSystem, stage, record)
    return case


def test_matrix_covers_every_asr_stage():
    assert _STAGES == [spec.name for spec in ASRSystem.stages]


@pytest.mark.parametrize("stage", _STAGES)
def test_directory_resolves_from_role_configs(case, stage):
    result = resolve_stage_configs(ASRSystem, [stage], case.configs, {})[0][stage]
    key = _CASES[stage]["directory_key"]
    assert OmegaConf.select(result, key) == str(case.get_stage_directory(stage))


@pytest.mark.parametrize("stage", _STAGES)
def test_stage_restores_from_saved_configs(case, stage):
    case.save_history(_STAGES[: _STAGES.index(stage) + 1])
    result = resolve_stage_configs(
        ASRSystem, [stage], {}, case.get_directory_overrides()
    )[0][stage]
    key = _CASES[stage]["directory_key"]
    assert OmegaConf.select(result, key) == str(case.get_stage_directory(stage))


@pytest.mark.parametrize("stage", _STAGES)
def test_cli_directory_overrides_role_config(case, stage):
    key = _CASES[stage]["directory_key"]
    target = case.root / "overridden"
    case.execute([stage], {key: target})
    assert OmegaConf.select(OmegaConf.create(case.observed[stage]), key) == str(target)
    assert case.calls == [stage]


@pytest.mark.parametrize("stage", _STAGES)
def test_unresolvable_required_directory_names_stage_and_key(case, stage):
    key = _CASES[stage]["directory_key"]
    configs = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(case.configs)))
    spec = next(s for s in ASRSystem.stages if s.name == stage)
    roles = (spec.config,) if isinstance(spec.config, str) else spec.config
    # The role contributing this stage's own setting must resolve it, regardless
    # of whether some earlier stage had a different value for the same key.
    OmegaConf.update(configs[roles[-1]], key, "${unresolved_directory}", merge=False)
    with pytest.raises(ConfigError, match=stage) as error:
        resolve_stage_configs(ASRSystem, [stage], configs, {})[0][stage]
    assert "unresolved_directory" in str(error.value)


@pytest.mark.parametrize("stage", _STAGES)
def test_single_stage_persists_only_itself(case, stage):
    case.execute([stage])
    assert case.calls == [stage]
    snapshots = [
        p for p in case.root.rglob("*.yaml") if p.name in {f"{s}.yaml" for s in _STAGES}
    ]
    assert snapshots
    assert {p.name for p in snapshots} == {f"{stage}.yaml"}


@pytest.mark.parametrize("stage", _STAGES)
def test_replay_preserves_preceding_snapshots_and_nonempty_own_config(case, stage):
    previous = _STAGES[: _STAGES.index(stage) + 1]
    case.save_history(previous)
    before = {
        p: p.read_bytes()
        for p in case.root.rglob("*.yaml")
        if p.name in {f"{s}.yaml" for s in previous if s != stage}
    }
    case.execute([stage], case.get_directory_overrides(), configs={})
    assert case.calls == [stage]
    for path, data in before.items():
        assert path.read_bytes() == data
    baked = case.exp / "config" / f"{stage}.yaml"
    assert OmegaConf.load(baked), "A config restored from a snapshot must not become {}"


@pytest.mark.parametrize(
    "stages",
    [
        _STAGES[:4],
        ["train", "infer"],
        ["infer", "measure"],
        ["pack_model", "upload_model"],
        ["pack_demo", "upload_demo"],
        _STAGES,
    ],
)
def test_stage_sequences_resolve_and_persist_in_declared_order(case, stages):
    case.execute(stages)
    assert case.calls == stages
    for stage in stages:
        assert (case.exp / "config" / f"{stage}.yaml").is_file()
    assert sorted(p.stem for p in (case.exp / "config").glob("*.yaml")) == sorted(
        stages
    )


@pytest.mark.parametrize("source", ["config", "snapshot", "cli", "missing", "unused"])
def test_train_resolves_stats_directory_from_any_available_source(case, source):
    training = OmegaConf.create(OmegaConf.to_container(case.configs["training"]))
    if source != "unused":
        training.normalization = {"stats_file": "${stats_dir}/feats_stats.npz"}
    overrides = {}
    if source != "config":
        del training.stats_dir
    if source == "snapshot":
        case.save_history(["collect_stats"])
        overrides["stats_dir"] = str(case.root / "stats")
    elif source == "cli":
        overrides["stats_dir"] = str(case.root / "stats")
    if source == "missing":
        with pytest.raises(ConfigError, match="stats_dir"):
            resolve_stage_configs(
                ASRSystem, ["train"], {"training": training}, overrides
            )[0]["train"]
        return
    result = resolve_stage_configs(
        ASRSystem, ["train"], {"training": training}, overrides
    )[0]["train"]
    if source != "unused":
        assert result.normalization.stats_file == str(
            case.root / "stats" / "feats_stats.npz"
        )


@pytest.mark.parametrize("source", ["config", "snapshot", "cli", "missing"])
def test_infer_resolves_training_directory_from_any_available_source(case, source):
    inference = OmegaConf.create(OmegaConf.to_container(case.configs["inference"]))
    inference.inference_dir = str(case.root / "decode")
    overrides = {}
    if source == "config":
        inference.exp_dir = str(case.exp)
    elif source in ("snapshot", "cli"):
        overrides["exp_dir"] = str(case.exp)
        if source == "snapshot":
            case.save_history(["train"])
    if source == "missing":
        with pytest.raises(ConfigError, match="exp_dir"):
            resolve_stage_configs(ASRSystem, ["infer"], {"inference": inference}, {})[
                0
            ]["infer"]
        return
    result = resolve_stage_configs(
        ASRSystem, ["infer"], {"inference": inference}, overrides
    )[0]["infer"]
    assert result.model.asr_model_file == str(case.exp / "valid.loss.best.pth")


@pytest.mark.parametrize("source", ["directory", "config"])
def test_measure_finds_inference_from_directory_or_config(case, source):
    case.save_history(["train", "infer"])
    configs = {"metrics": case.configs["metrics"]}
    overrides = {"exp_dir": str(case.exp)}
    if source == "directory":
        overrides["inference_dir"] = str(case.exp / "inference")
    else:
        configs["inference"] = case.configs["inference"]
    result = resolve_stage_configs(ASRSystem, ["measure"], configs, overrides)[0][
        "measure"
    ]
    assert result.score_file == str(case.exp / "inference" / "test" / "text.scp")


def test_later_unresolvable_demo_config_does_not_block_train(case):
    case.configs["demo"].pack.out_dir = "${missing_demo_directory}"
    result = resolve_stage_configs(ASRSystem, ["train"], case.configs, {})[0]["train"]
    assert result.exp_dir == str(case.exp)


def test_measure_uses_selected_inference_run_without_mixing_another(case):
    case.save_history(["train", "infer"])
    other = case.root / "another_decode"
    other.mkdir()
    config = case.load_snapshot("infer")
    config.inference_dir = str(other)
    config.model.asr_model_file = "another-model.pth"
    OmegaConf.save(config, other / "infer.yaml")
    result = resolve_stage_configs(
        ASRSystem,
        ["measure"],
        {"metrics": case.configs["metrics"]},
        {"inference_dir": str(other)},
    )[0]["measure"]
    assert result.model.asr_model_file == "another-model.pth"
    assert result.score_file == str(other / "test" / "text.scp")


@pytest.mark.parametrize("stage", _STAGES)
def test_cli_paths_are_saved_as_portable_yaml_strings(case, stage):
    key = _CASES[stage]["directory_key"]
    case.execute([stage], {key: case.root / "selected"})
    saved = [p for p in case.root.rglob(f"{stage}.yaml") if "history" not in p.parts]
    assert saved
    for path in saved:
        text = path.read_text()
        assert "!!python" not in text
        config = OmegaConf.load(path)
        value = OmegaConf.select(config, key)
        if value is not None:
            assert isinstance(value, str)


@pytest.mark.parametrize("directory", [None, ""])
@pytest.mark.parametrize(
    "stage,key,role",
    [
        ("train", "stats_dir", "training"),
        ("infer", "exp_dir", "inference"),
    ],
)
def test_required_dependency_directory_must_not_be_null_or_empty(
    case, stage, key, role, directory
):
    config = OmegaConf.create(OmegaConf.to_container(case.configs[role]))
    OmegaConf.update(config, key, directory, merge=False)
    if stage == "train":
        config.normalization = {"stats_file": "${stats_dir}/feats_stats.npz"}
    else:
        config.inference_dir = str(case.root / "decode")
    with pytest.raises(ConfigError, match=key):
        resolve_stage_configs(ASRSystem, [stage], {role: config}, {})[0][stage]


@pytest.mark.parametrize(
    "stage,key,role,hint",
    [
        ("train", "stats_dir", "training", "--stats-dir"),
        ("infer", "exp_dir", "inference", "--exp"),
    ],
)
def test_missing_dependency_directory_error_explains_how_to_supply_it(
    case, stage, key, role, hint
):
    config = OmegaConf.create(OmegaConf.to_container(case.configs[role]))
    if key in config:
        del config[key]
    if stage == "train":
        config.normalization = {"stats_file": "${stats_dir}/feats_stats.npz"}
    else:
        config.inference_dir = str(case.root / "decode")
    with pytest.raises(ConfigError) as error:
        resolve_stage_configs(ASRSystem, [stage], {role: config}, {})[0][stage]
    assert stage in str(error.value)
    assert key in str(error.value)
    assert hint in str(error.value)


def test_train_can_reuse_external_stats_from_another_dataset(case):
    case.save_history(["create_dataset"])
    shared = case.root / "shared_stats"
    shared.mkdir()
    previous = case.load_snapshot("collect_stats")
    previous.exp_dir = str(case.root / "another_experiment")
    previous.data_dir = str(case.root / "another_dataset")
    previous.stats_dir = str(shared)
    OmegaConf.save(previous, shared / "collect_stats.yaml")
    training = OmegaConf.create(OmegaConf.to_container(case.configs["training"]))
    training.normalization = {"stats_file": "${stats_dir}/feats_stats.npz"}
    result = resolve_stage_configs(
        ASRSystem,
        ["train"],
        {"training": training},
        {"exp_dir": str(case.exp), "stats_dir": str(shared)},
    )[0]["train"]
    assert result.data_dir == str(case.root / "data")
    assert result.normalization.stats_file == str(shared / "feats_stats.npz")


def test_saved_parent_directory_can_use_an_equivalent_relative_path(case):
    case.save_history(["train", "infer"])
    inference = case.load_snapshot("infer")
    inference.exp_dir = os.path.relpath(case.exp, Path.cwd())
    OmegaConf.save(inference, case.get_stage_directory("infer") / "infer.yaml")
    result = resolve_stage_configs(
        ASRSystem, ["infer"], {}, case.get_directory_overrides()
    )[0]["infer"]
    assert result.exp_dir == str(case.exp)
