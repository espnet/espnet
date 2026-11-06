"""Config inheritance across a system's stages: inherit, bake, and reload.

Stage configs inherit earlier settings in declaration order by whole-key
replacement. Executed stages persist full snapshots at their declared log
directories and own-key records under <exp_dir>/config/. Reading a preceding
stage never runs or persists it; no separate experiment-context file is used.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError, MissingMandatoryValue

from espnet3.contract.stages import (
    resolve_config_roles,
    resolve_stage_log_dir,
)
from espnet3.utils.config_utils import convert_to_dict
from espnet3.utils.run_utils import ConfigError, resolve_loaded_configs
from espnet3.utils.stage_cli import build_stage_flag_parser

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
    requested = list(upto) if upto is not None else [s.name for s in system_cls.stages]
    configs, own_keys, _ = resolve_stage_configs(
        system_cls, requested, configs_by_role, {}, search_root=exp_dir
    )
    return configs, own_keys


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


def snapshot_config(
    log_dir,
    stage,
    resolved,
    *,
    dry_run=False,
    is_rank0=True,
):
    """Persist the full resolved config for the stage at <log_dir>/<stage>.yaml.

    - Full effective config persisted (not just own_keys)
    - Atomic writes with history backup when content changes
    - Respects dry_run and rank-0-only
    """
    if dry_run:
        logger.debug("dry_run: skipping snapshot write for stage %s", stage)
        return
    if not is_rank0:
        logger.debug(
            "rank!=0: skipping snapshot write for stage %s in distributed execution",
            stage,
        )
        return
    if not log_dir:
        logger.debug("log_dir not set for stage %s: skipping snapshot write", stage)
        return

    import os
    from pathlib import Path

    snapshot_path = Path(log_dir) / f"{stage}.yaml"
    try:
        new_yaml = OmegaConf.to_yaml(resolved, resolve=True)
    except Exception as exc:
        logger.warning(
            "Failed to serialize resolved config for stage %s: %s", stage, exc
        )
        return

    backup_needed = snapshot_path.exists()
    if backup_needed:
        try:
            existing = OmegaConf.load(snapshot_path)
            if OmegaConf.to_container(
                existing, resolve=False
            ) == OmegaConf.to_container(resolved, resolve=False):
                backup_needed = False
        except Exception:
            pass

    if backup_needed and snapshot_path.exists():
        history_dir = Path(log_dir) / ".config_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        import time

        timestamp = int(time.time())
        history_path = history_dir / f"{stage}.{timestamp}.yaml"
        shutil.copy2(str(snapshot_path), str(history_path))
        logger.info(
            "%s: config snapshot differs from previous run; backed up history to %s",
            stage,
            history_path,
        )

    tmp_path = snapshot_path.with_suffix(".yaml.tmp")
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(new_yaml, encoding="utf-8")
        os.replace(str(tmp_path), str(snapshot_path))
        logger.debug("Wrote snapshot: %s", snapshot_path)
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        logger.error("Failed to write snapshot for stage %s: %s", stage, exc)
        raise


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


def _in_scope_stages(system_cls: type, requested_stage: str) -> Tuple:
    """Return `system_cls.stages` up to and including `requested_stage`.

    Raises:
        ConfigError: `requested_stage` is not declared on `system_cls`.
    """
    names = [spec.name for spec in system_cls.stages]
    if requested_stage not in names:
        raise ConfigError(
            f"stage {requested_stage!r} is not declared on "
            f"{system_cls.__name__}.stages"
        )
    return system_cls.stages[: names.index(requested_stage) + 1]


def resolve_search_directories(
    system_cls: type,
    requested_stage: str,
    configs_by_role: Mapping[str, Optional[DictConfig]],
    cli_dir_overrides: Mapping[str, "str | Path"],
    visited: Optional[set] = None,
) -> Dict[str, Path]:
    """Directory to search for each in-scope stage's saved config snapshot.

    Only the requested stage and stages declared before it are considered -
    a later, unrelated stage's unresolvable `log_dir` never raises or
    appears in the result. For each in-scope stage: a CLI directory
    override for its canonical `log_dir` key wins; otherwise, if this run's
    own config for that stage can already resolve `log_dir` (via
    :func:`~espnet3.contract.stages.resolve_stage_log_dir`), that value is
    used. A stage with neither is simply omitted from the result - this is
    not an error, since a directory may still be discovered later from a
    loaded snapshot (see :func:`resolve_stage_configs`).

    Args:
        system_cls: The system class whose declared `stages` sets the
            inheritance order.
        requested_stage: The stage whose saved config search this is for.
        configs_by_role: This run's own, not-yet-resolved config for each
            role, as in :func:`build_stage_configs`.
        cli_dir_overrides: Canonical `log_dir` key (e.g. `"exp_dir"`,
            not an alias - alias-to-canonical translation is the CLI
            parser's job) -> directory explicitly given this run.
        visited: Optional set the caller may reuse across calls; populated
            with `(stage_name, directory)` pairs this call resolved.

    Returns:
        Stage name -> directory to search, for every in-scope stage that
        resolved to one.
    """
    if visited is None:
        visited = set()

    resolved: Dict[str, Path] = {}
    for spec in _in_scope_stages(system_cls, requested_stage):
        directory: Optional[Path] = None
        if spec.log_dir and spec.log_dir in cli_dir_overrides:
            directory = Path(cli_dir_overrides[spec.log_dir])
        else:
            own = _own_config_for(spec, configs_by_role)
            if own is not None:
                try:
                    directory = resolve_stage_log_dir(own, spec)
                except (InterpolationResolutionError, MissingMandatoryValue):
                    # Discovery precedes inheritance: this role may need a key
                    # from a prior stage. Validate it in the final merged config.
                    directory = None
        if directory is not None:
            resolved[spec.name] = directory
            visited.add((spec.name, directory))
    return resolved


def _resolve_config(stage, config, directory_keys, directory_help):
    """Resolve a stage, rejecting empty directory dependencies before expansion.

    Args:
        stage: Stage name included in configuration errors.
        config: Effective config, resolved in place.
        directory_keys: Directory keys declared by the system's StageSpec entries.
        directory_help: Help generated by the existing stage argument parser.

    Raises:
        ConfigError: A referenced directory is empty or an interpolation fails.

    Examples:
        >>> config = OmegaConf.create({"work_dir": "/tmp/job", "file": "${work_dir}/x"})
        >>> _resolve_config("example", config, {"work_dir"}, "--work-dir DIR")
        >>> config.file
        '/tmp/job/x'
    """

    def find_references(value):
        """Yield interpolation keys from unresolved config containers.

        Examples:
            >>> list(find_references({"file": "${work_dir}/x"}))
            ['work_dir']
        """
        if isinstance(value, dict):
            for item in value.values():
                yield from find_references(item)
        elif isinstance(value, list):
            for item in value:
                yield from find_references(item)
        elif isinstance(value, str):
            yield from re.findall(r"\$\{([\w.]+)\}", value)

    # Normalize only declared directory values; arbitrary user objects are not
    # silently converted. CLI values are strings before entering this resolver.
    for key in directory_keys:
        try:
            value = OmegaConf.select(config, key)
        except (InterpolationResolutionError, MissingMandatoryValue):
            continue
        if isinstance(value, Path):
            OmegaConf.update(config, key, str(value), merge=False)
    dependencies = (
        set(find_references(convert_to_dict(config, resolve=False))) & directory_keys
    )
    for key in sorted(dependencies):
        try:
            value = OmegaConf.select(config, key)
        except (InterpolationResolutionError, MissingMandatoryValue) as exc:
            raise ConfigError(
                f"{stage} config cannot resolve required directory {key!r}: "
                f"{exc}; set it in the config, inherit a saved config, or "
                f"use a directory option:\n{directory_help}"
            ) from exc
        if value is None or value == "":
            raise ConfigError(
                f"{stage} config cannot resolve required directory {key!r}; "
                f"set it in the config, inherit a saved config, or use "
                f"a directory option:\n{directory_help}"
            )
    try:
        resolve_loaded_configs({stage: config})
    except ConfigError as exc:
        if dependencies:
            raise ConfigError(f"{exc}\n{directory_help}") from exc
        raise


def resolve_stage_configs(
    system_cls: type,
    requested_stages: Iterable[str],
    configs_by_role: Mapping[str, Optional[DictConfig]],
    cli_dir_overrides: Mapping[str, "str | Path"],
    *,
    search_root: Optional[Path] = None,
    visited: Optional[set] = None,
) -> Tuple[Dict[str, DictConfig], Dict[str, Tuple[str, ...]], set[str]]:
    """Resolve stage configs and persistence keys without writing snapshots.

    Use stage-order whole-key merging; explicit configs replace their own
    snapshots, CLI directories win, and missing predecessors only warn.

    Args:
        system_cls: System defining stage order, roles and directory keys.
        requested_stages: Stages to run; only their predecessors are included.
        configs_by_role: Unresolved role configs with defaults already merged.
        cli_dir_overrides: Canonical directory keys and CLI values.
        search_root: Legacy snapshot search root, not a config override.
        visited: Optional set populated with loaded stage/directory pairs.

    Returns:
        (stage_configs, own_keys, available): configs, persistence keys, sources.

    Raises:
        ConfigError: Invalid stage/snapshot, conflicting parent, or unresolved input.

    Examples:
        >>> from espnet3.contract.stages import StageSpec
        >>> class Example:
        ...     stages = (StageSpec(name="run", config="job"),)
        >>> configs = {"job": OmegaConf.create({"seed": 1})}
        >>> resolve_stage_configs(Example, ["run"], configs, {})[1]
        {'run': ('seed',)}
    """
    requested = set(requested_stages)
    if not requested:
        return {}, {}, set()
    names = [spec.name for spec in system_cls.stages]
    for stage in requested:
        _in_scope_stages(system_cls, stage)
    specs = system_cls.stages[: max(names.index(stage) for stage in requested) + 1]
    overrides = {
        key: str(value) for key, value in cli_dir_overrides.items() if value is not None
    }
    directory_keys = {spec.log_dir for spec in system_cls.stages if spec.log_dir}
    directory_help = build_stage_flag_parser(system_cls.stages).format_help()
    if visited is None:
        visited = set()
    own = {spec.name: _own_config_for(spec, configs_by_role) for spec in specs}
    loaded = {}
    loaded_paths = {}
    directories = {}

    def assemble():
        running = {}
        result = {}
        keys = {}
        available = set()
        for spec in specs:
            contribution = own[spec.name]
            # Optional preceding stages represent the saved run, when present;
            # executing stages use this run's explicitly supplied settings.
            if spec.name not in requested and spec.name in loaded:
                contribution = loaded[spec.name]
            elif contribution is None:
                contribution = loaded.get(spec.name)
            if contribution is not None:
                payload = convert_to_dict(contribution, resolve=False)
                available.add(spec.name)
                running.update(payload)
                keys[spec.name] = tuple(payload)
            else:
                keys[spec.name] = ()
            current = OmegaConf.create(running)
            for key, value in overrides.items():
                OmegaConf.update(current, key, value, merge=False)
            # Freeze earlier resolved values before later stages override keys
            # they depended on. Unresolved optional inputs can be discovered on
            # a later pass, and requested stages are validated below.
            try:
                _resolve_config(spec.name, current, directory_keys, directory_help)
            except ConfigError:
                pass
            running = convert_to_dict(current, resolve=False)
            result[spec.name] = current
        return result, keys, available

    # A full snapshot already contains the preceding configuration, including
    # all directory paths. Discover it first, then read those known locations
    # once to retain each preceding stage's original settings.
    for use_snapshot_directories in (False, True):
        partial, _, _ = assemble()
        # Include raw directory hints provided by later in-scope roles: a train
        # config can name an earlier statistics directory without a stats role.
        context = {}
        for spec in specs:
            if use_snapshot_directories and spec.name in loaded:
                context.update(convert_to_dict(loaded[spec.name], resolve=False))
            if own[spec.name] is not None:
                context.update(convert_to_dict(own[spec.name], resolve=False))
        context_config = OmegaConf.create(context)
        for key, value in overrides.items():
            OmegaConf.update(context_config, key, value, merge=False)
        roots = []
        if search_root is not None:
            roots.append(Path(search_root))
        for config in [*partial.values(), context_config]:
            try:
                root = OmegaConf.select(config, "exp_dir")
                if root and Path(root) not in roots:
                    roots.append(Path(root))
            except (InterpolationResolutionError, MissingMandatoryValue):
                pass
        for spec in specs:
            directory = None
            if spec.log_dir:
                for config in [partial[spec.name], context_config]:
                    try:
                        value = OmegaConf.select(config, spec.log_dir)
                    except (InterpolationResolutionError, MissingMandatoryValue):
                        continue
                    if value:
                        directory = Path(value)
                        break
            candidates = []
            if directory is not None:
                directories[spec.name] = directory
                candidates.append(directory / f"{spec.name}.yaml")
            # A full snapshot may use exp_dir as the fallback for log_dir=None.
            for root in roots:
                if not spec.log_dir:
                    candidates.append(root / f"{spec.name}.yaml")
            for root in roots:
                candidates.append(root / _CONFIG_DIRNAME / f"{spec.name}.yaml")
            path = next((p for p in candidates if p.is_file()), None)
            if path is not None and loaded_paths.get(spec.name) != path:
                try:
                    snapshot = OmegaConf.load(path)
                    if not isinstance(snapshot, DictConfig):
                        raise ValueError("expected a configuration mapping")
                except Exception as exc:
                    raise ConfigError(
                        f"stage {spec.name!r}: could not read saved config "
                        f"snapshot at {path}: {exc}"
                    ) from exc
                loaded[spec.name] = snapshot
                loaded_paths[spec.name] = path
                visited.add((spec.name, path.parent))

    # Catch snapshots from different training runs, for example:
    #   --exp_dir exp/model_B --inference-dir exp/model_A/inference
    # model_B/train.yaml records exp_dir=exp/model_B, but the saved infer.yaml
    # records exp_dir=exp/model_A. Reject this mismatch before CLI overrides
    # hide it. Check only exp_dir: data_dir and stats_dir may be shared across
    # training runs.
    for index, spec in enumerate(specs):
        snapshot = loaded.get(spec.name)
        if snapshot is None or (spec.name in requested and own[spec.name] is not None):
            continue
        for prior in specs[:index]:
            if prior.log_dir != "exp_dir" or prior.log_dir == spec.log_dir:
                continue
            parent = loaded.get(prior.name)
            if parent is None:
                continue
            left = OmegaConf.select(parent, prior.log_dir)
            right = OmegaConf.select(snapshot, prior.log_dir)
            if left and right and Path(left).resolve() != Path(right).resolve():
                raise ConfigError(
                    f"conflicting saved configs: stage {spec.name!r} and "
                    f"stage {prior.name!r} disagree on directory "
                    f"{prior.log_dir!r} ({right!r} vs {left!r})"
                )

    stage_configs, own_keys, available = assemble()
    for spec in specs:
        config = stage_configs[spec.name]
        if spec.name in requested:
            _resolve_config(spec.name, config, directory_keys, directory_help)
        elif spec.name in available:
            try:
                _resolve_config(spec.name, config, directory_keys, directory_help)
            except ConfigError:
                # An unexecuted stage may need inputs supplied only to a later
                # stage. Its unresolved fields must not block that later stage.
                logger.warning(
                    "Preceding stage %r cannot resolve all settings yet; "
                    "validating the requested stages instead.",
                    spec.name,
                )
        elif spec.name not in requested:
            logger.warning(
                "Preceding stage %r has no saved or supplied config; continuing "
                "without inheriting its own settings.",
                spec.name,
            )
        # CLI values are part of this run's own settings, including nested keys.
        if spec.name in requested:
            own_keys[spec.name] = tuple(
                dict.fromkeys(
                    [
                        *own_keys[spec.name],
                        *(key.split(".")[0] for key in overrides),
                    ]
                )
            )
    return stage_configs, own_keys, available
