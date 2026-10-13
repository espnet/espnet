"""Config inheritance across a system's stages: inherit, bake, and reload.

Each stage's config is the earlier stages' baked, top-level keys (stage
order, whole-key replacement, no deep merge), with the stage's own config on
top. The only persisted record of what a stage actually ran with is the
`<exp_dir>/config/<stage>.yaml` that stage's own run bakes; there is no
separate experiment-context file.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from espnet3.contract.stages import resolve_config_roles
from espnet3.utils.run_utils import ConfigError, resolve_loaded_configs

logger = logging.getLogger(__name__)

_CONFIG_DIRNAME = "config"
_HISTORY_DIRNAME = "history"


def _history_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def inherit(
    exp_dir: Optional[Path],
    system_cls: type,
    stage: str,
    own: DictConfig,
    *,
    in_memory: Optional[Mapping[str, Mapping]] = None,
) -> DictConfig:
    """Earlier stages' configs, in stage order, then `own`'s keys on top.

    Walks `system_cls.stages` in declared order up to (not including)
    `stage`. For each earlier stage, its top-level keys come from
    `in_memory[stage_name]` when given (the same run's own keys, already
    resolved, before that stage has baked anything), else from
    `<exp_dir>/config/<stage_name>.yaml` when that file exists (a `--exp_dir`
    run inheriting from an earlier run). A later stage's keys win over an
    earlier one's, and `own` wins over all of them. The merge is
    whole-top-level-key replacement, not a deep merge: a key present in
    `own` always fully replaces the same key inherited from an earlier
    stage.

    Args:
        exp_dir: The experiment directory holding `config/<stage>.yaml`
            for earlier stages, or `None` when none of them has one (no
            `in_memory` entry and no baked file for any earlier stage).
        system_cls: The system class `stage` belongs to; its declared
            `stages` gives the inheritance order.
        stage: The stage about to run. Only stages declared before it are
            considered; `stage` itself contributes nothing inherited.
        own: This stage's own config (its default plus whatever the
            caller loaded), not yet merged with anything inherited.
        in_memory: Mapping from an earlier stage's name to its own
            top-level keys, already resolved, for a stage that has not
            (yet) baked a file this run. Checked before the baked file.

    Returns:
        DictConfig: A new config: earlier stages' keys (`in_memory` or
        baked file), then `own`'s own keys layered on top.

    Examples:
        >>> import tempfile
        >>> from espnet3.contract.stages import StageSpec
        >>> class ExampleSystem:
        ...     stages = (
        ...         StageSpec(name="train", config="training"),
        ...         StageSpec(name="infer", config="inference"),
        ...     )
        >>> with tempfile.TemporaryDirectory() as exp_dir:
        ...     config_dir = Path(exp_dir) / "config"
        ...     config_dir.mkdir()
        ...     _ = (config_dir / "train.yaml").write_text("exp_dir: " + exp_dir)
        ...     own = OmegaConf.create({"inference_dir": "infer"})
        ...     merged = inherit(exp_dir, ExampleSystem, "infer", own)
        ...     merged.exp_dir == exp_dir and merged.inference_dir == "infer"
        True
        >>> own2 = OmegaConf.create({"inference_dir": "infer2"})
        >>> merged2 = inherit(
        ...     None, ExampleSystem, "infer", own2,
        ...     in_memory={"train": {"exp_dir": "./exp/live"}},
        ... )
        >>> merged2.exp_dir
        './exp/live'
    """
    in_memory = in_memory or {}
    merged: dict = {}
    for spec in system_cls.stages:
        if spec.name == stage:
            break
        if spec.name in in_memory:
            merged.update(dict(in_memory[spec.name]))
            continue
        if exp_dir is None:
            continue
        path = Path(exp_dir) / _CONFIG_DIRNAME / f"{spec.name}.yaml"
        if path.exists():
            baked = OmegaConf.to_container(
                load_baked(exp_dir, spec.name), resolve=False
            )
            merged.update(baked)
    own_dict = OmegaConf.to_container(own, resolve=False) if own is not None else {}
    merged.update(own_dict)
    return OmegaConf.create(merged)


def _own_config_for(
    spec, configs_by_role: Mapping[str, Optional[DictConfig]]
) -> Optional[DictConfig]:
    """Return a stage's own config: its single role's, or all given roles merged.

    `StageSpec.config` is usually a single role, and this is just
    `configs_by_role.get(spec.config)`. When it names more than one role
    (e.g. `("training", "publication")`), the given roles' own configs are
    merged (whole-top-level-key replacement, later role wins), so a stage
    needing more than one role still gets a single config to run with -
    the other roles' keys still reach it the normal way too, via
    `inherit`'s walk of earlier stages, so this only matters for a role
    that is not also an earlier stage's own role.
    """
    parts = [configs_by_role.get(role) for role in resolve_config_roles(spec.config)]
    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    merged: dict = {}
    for part in parts:
        merged.update(OmegaConf.to_container(part, resolve=False))
    return OmegaConf.create(merged)


def build_stage_configs(
    system_cls: type,
    configs_by_role: Mapping[str, Optional[DictConfig]],
    *,
    exp_dir: Optional[Path],
    upto: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, DictConfig], Dict[str, Tuple[str, ...]]]:
    """Resolve each requested stage's config, inheriting from the earlier ones.

    For each stage in `system_cls.stages`, up to and including the last
    stage named in `upto` (all of them when `upto` is `None`): merges the
    stage's own config (from `configs_by_role`, by its declared role) on
    top of the earlier stages' own keys via :func:`inherit`, resolves the
    result, then records its own keys' resolved values as the in-memory
    equivalent of what that stage would bake - so a later stage in the
    same call inherits from it without needing a file on disk yet.

    Args:
        system_cls: The system class whose declared `stages` sets the
            inheritance order.
        configs_by_role: Each config role's own config (as loaded, not
            yet resolved), or `None` for a role nothing was given for.
        exp_dir: Passed to :func:`inherit` as the fallback for a stage not
            covered by this call's own in-memory results (an earlier run's
            baked file). `None` when there is none.
        upto: Stage names actually requested this run. Stages declared
            after all of them are not resolved at all, so an unrelated
            later stage's missing identity never stops this call.

    Returns:
        A pair of mappings, both keyed by stage name: the resolved config
        each stage should run with, and the top-level keys that stage's
        own config (as opposed to what it inherited) contributed.

    Raises:
        ConfigError: A stage's merged config has an unresolved
            interpolation; see :func:`~espnet3.utils.run_utils.resolve_loaded_configs`.

    Examples:
        >>> from espnet3.contract.stages import StageSpec
        >>> class ExampleSystem:
        ...     stages = (
        ...         StageSpec(name="train", config="training"),
        ...         StageSpec(name="infer", config="inference"),
        ...     )
        >>> configs = {
        ...     "training": OmegaConf.create(
        ...         {"exp_tag": "t", "exp_dir": "./exp/${exp_tag}"}
        ...     ),
        ...     "inference": OmegaConf.create(
        ...         {"inference_dir": "${exp_dir}/infer"}
        ...     ),
        ... }
        >>> stage_configs, own_keys = build_stage_configs(
        ...     ExampleSystem, configs, exp_dir=None
        ... )
        >>> stage_configs["infer"].inference_dir
        './exp/t/infer'
        >>> own_keys["train"]
        ('exp_tag', 'exp_dir')
    """
    if upto is not None:
        upto = list(upto)
        stage_names = [s.name for s in system_cls.stages]
        last_index = max(stage_names.index(name) for name in upto)
        stages = system_cls.stages[: last_index + 1]
    else:
        stages = system_cls.stages

    in_memory: Dict[str, Dict] = {}
    stage_configs: Dict[str, DictConfig] = {}
    own_keys: Dict[str, Tuple[str, ...]] = {}

    for spec in stages:
        own = _own_config_for(spec, configs_by_role)
        this_own_keys = tuple(own.keys()) if own is not None else ()
        merged = inherit(
            exp_dir,
            system_cls,
            spec.name,
            own if own is not None else OmegaConf.create({}),
            in_memory=in_memory,
        )
        resolve_loaded_configs({spec.name: merged})
        stage_configs[spec.name] = merged
        own_keys[spec.name] = this_own_keys
        in_memory[spec.name] = {key: merged[key] for key in this_own_keys}

    return stage_configs, own_keys


def bake(
    exp_dir: Path, stage: str, resolved: DictConfig, own_keys: Iterable[str]
) -> Path:
    """Write `<exp_dir>/config/<stage>.yaml`: `stage`'s own resolved keys.

    Only `own_keys` of `resolved` are written - the keys `stage`'s own
    config contributed, not the ones it inherited from an earlier stage's
    bake - so re-running an unrelated later stage never rewrites what an
    earlier stage already baked. The write is atomic (temp file, then
    rename), so a run that fails partway through never leaves a corrupt
    `<stage>.yaml` for a later run to inherit.

    Args:
        exp_dir: The experiment directory; `config/` is created if missing.
        stage: The stage name whose resolved keys are being baked.
        resolved: The fully merged, resolved config `stage` is about to
            run with.
        own_keys: The top-level keys of `resolved` that are `stage`'s own
            (as opposed to inherited) - typically the keys of the config
            loaded for `stage` before `inherit` was applied.

    Returns:
        Path: The written `<stage>.yaml` path.

    Examples:
        >>> import tempfile
        >>> from omegaconf import OmegaConf
        >>> resolved = OmegaConf.create({"exp_dir": "./exp", "seed": 1})
        >>> with tempfile.TemporaryDirectory() as exp_dir:
        ...     path = bake(Path(exp_dir), "train", resolved, ["seed"])
        ...     OmegaConf.load(path)
        {'seed': 1}
    """
    config_dir = Path(exp_dir) / _CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    stage_path = config_dir / f"{stage}.yaml"

    payload = OmegaConf.create({key: resolved[key] for key in own_keys})
    new_text = OmegaConf.to_yaml(payload, resolve=True)

    if stage_path.exists():
        old_text = stage_path.read_text(encoding="utf-8")
        if old_text == new_text:
            return stage_path  # Unchanged since the last run; leave as-is.
        history_dir = config_dir / _HISTORY_DIRNAME
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{stage}.{_history_timestamp()}.yaml"
        shutil.move(str(stage_path), str(history_path))
        logger.warning(
            "%s: config/%s.yaml differs from the last run; the previous "
            "one is kept as %s",
            stage,
            stage,
            history_path,
        )

    tmp_path = stage_path.with_suffix(".yaml.tmp")
    tmp_path.write_text(new_text, encoding="utf-8")
    os.replace(tmp_path, stage_path)
    return stage_path


def load_baked(exp_dir: Path, stage: str) -> DictConfig:
    """Read `<exp_dir>/config/<stage>.yaml`.

    Args:
        exp_dir: The experiment directory `stage` was (expected to be) run
            under.
        stage: The stage whose baked config to read.

    Returns:
        DictConfig: The config `stage` ran with, as it was baked - its own
        top-level keys only, already resolved.

    Raises:
        ConfigError: No `<stage>.yaml` exists under `exp_dir`; the message
            names `stage` and the path that was checked.

    Examples:
        >>> import tempfile
        >>> from omegaconf import OmegaConf
        >>> with tempfile.TemporaryDirectory() as exp_dir:
        ...     resolved = OmegaConf.create({"seed": 1})
        ...     path = bake(Path(exp_dir), "train", resolved, ["seed"])
        ...     load_baked(Path(exp_dir), "train")
        {'seed': 1}
    """
    path = Path(exp_dir) / _CONFIG_DIRNAME / f"{stage}.yaml"
    if not path.is_file():
        raise ConfigError(f"no baked config for stage {stage!r} under {path}")
    return OmegaConf.load(path)
