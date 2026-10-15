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

from espnet3.contract.stages import resolve_config_roles, resolve_stage_log_dir
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
    loaded snapshot (see :func:`merge_stage_config_from_snapshots`).

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
                directory = resolve_stage_log_dir(own, spec)
        if directory is not None:
            resolved[spec.name] = directory
            visited.add((spec.name, directory))
    return resolved


def merge_stage_config_from_snapshots(
    system_cls: type,
    requested_stage: str,
    configs_by_role: Mapping[str, Optional[DictConfig]],
    cli_dir_overrides: Mapping[str, "str | Path"],
    *,
    visited: Optional[set] = None,
) -> DictConfig:
    """Build `requested_stage`'s config from saved snapshots and this run.

    Implements design/config_tree_fix.md's "保存済み config の探索" (steps
    5-8) and "設定の優先順位と独立性". For each in-scope stage (the
    requested stage and the ones declared before it), a directory to search
    is found via :func:`resolve_search_directories`, extended with any
    directory a loaded snapshot itself reveals for another in-scope stage's
    `log_dir` key (bounded by the number of in-scope stages; `visited`
    guards against reloading the same `(stage, directory)` pair twice).

    Each in-scope stage's config is then built in declared order, lowest to
    highest priority:

    1. The preceding in-scope stages' own already-built configs, layered in
       declared order via whole-top-level-key replacement (never a deep
       merge) - the same rule :func:`inherit` uses.
    2. This stage's own previously-saved snapshot, *only* when this run did
       not explicitly give this stage's own role config (an explicit config
       always wins over - and never gets topped up by - a stale snapshot,
       so a setting deleted from the new config cannot resurrect itself
       from an old one).
    3. This run's explicitly-given role config for this stage.

    `cli_dir_overrides` is applied last, as dotted-key updates into the
    *requested* stage's final config only (its own nested key, not a flat
    top-level replace) - it is this run's final override of where to read
    or write, not a value for every stage's config.

    Every stage's result is an independent copy: later mutation of one
    stage's dict can never leak into another's.

    Args:
        system_cls: The system class whose declared `stages` sets order.
        requested_stage: The stage whose config to build.
        configs_by_role: This run's own, not-yet-resolved config for each
            role (`None` for a role nothing was given for this run).
        cli_dir_overrides: Canonical `log_dir` key -> directory/value this
            run's CLI explicitly gave. Dotted keys apply to the nested key.
        visited: Optional set the caller may reuse across calls.

    Returns:
        DictConfig: `requested_stage`'s resolved config.

    Raises:
        ConfigError: `requested_stage` is not declared; a snapshot file
            exists but cannot be parsed; two loaded snapshots disagree on
            the value of a top-level key they both define (a sign of two
            different parent experiments being mixed); or a required
            interpolation in the final config cannot be resolved.
    """
    if visited is None:
        visited = set()

    in_scope = _in_scope_stages(system_cls, requested_stage)
    directories = resolve_search_directories(
        system_cls,
        requested_stage,
        configs_by_role,
        cli_dir_overrides,
        visited=visited,
    )

    # Load each in-scope stage's snapshot where a directory is known,
    # expanding `directories` with whatever a loaded snapshot reveals about
    # another in-scope stage's own log_dir key. Bounded by len(in_scope):
    # each pass can only newly resolve one more stage's directory.
    #
    # `attempted` tracks load attempts specifically - distinct from
    # `visited`, which resolve_search_directories already populated for
    # every stage/directory pair it *resolved* (not attempted to load), so
    # reusing `visited` here would wrongly skip loading every stage on the
    # very first pass.
    attempted: set = set()
    loaded: Dict[str, DictConfig] = {}

    for _ in range(len(in_scope) + 1):
        progressed = False
        for spec in in_scope:
            if spec.name in loaded:
                continue
            directory = directories.get(spec.name)
            if directory is None:
                continue
            key = (spec.name, directory)
            if key in attempted:
                continue
            attempted.add(key)
            visited.add(key)
            snapshot_path = Path(directory) / f"{spec.name}.yaml"
            if not snapshot_path.is_file():
                continue
            try:
                loaded[spec.name] = OmegaConf.load(snapshot_path)
            except Exception as exc:
                raise ConfigError(
                    f"stage {spec.name!r}: could not read saved config "
                    f"snapshot at {snapshot_path}: {exc}"
                ) from exc
            progressed = True
            for other in in_scope:
                if other.name in directories or other.name in loaded:
                    continue
                if not other.log_dir:
                    continue
                value = OmegaConf.select(loaded[spec.name], other.log_dir)
                if value:
                    directories[other.name] = Path(value)
        if not progressed:
            break

    # Narrow fallback for the requested stage only, used when the normal
    # search found nothing for it (e.g. it declares no log_dir, so
    # resolve_search_directories has nothing to resolve) and this run gave
    # an explicit exp_dir. New-style <exp_dir>/<stage>.yaml is tried first -
    # a log_dir=None stage still snapshots to exp_dir (see "保存・ログ・
    # fallback") - and only the OLD <exp_dir>/config/<stage>.yaml layout
    # after that, so a new snapshot always wins over an old one.
    if requested_stage not in loaded and "exp_dir" in cli_dir_overrides:
        exp_dir_path = Path(cli_dir_overrides["exp_dir"])
        new_style_path = exp_dir_path / f"{requested_stage}.yaml"
        old_config_path = exp_dir_path / _CONFIG_DIRNAME / f"{requested_stage}.yaml"
        if new_style_path.is_file():
            try:
                loaded[requested_stage] = OmegaConf.load(new_style_path)
                directories[requested_stage] = exp_dir_path
            except Exception as exc:
                raise ConfigError(
                    f"stage {requested_stage!r}: could not read saved config "
                    f"snapshot at {new_style_path}: {exc}"
                ) from exc
        elif old_config_path.is_file():
            try:
                loaded[requested_stage] = load_baked(exp_dir_path, requested_stage)
                directories[requested_stage] = exp_dir_path
            except ConfigError:
                pass

    # Detect snapshots that disagree on a key they both define - mixing two
    # different parent experiments rather than one shared lineage.
    seen_values: Dict[str, Tuple[str, object]] = {}
    for spec in in_scope:
        snapshot = loaded.get(spec.name)
        if snapshot is None:
            continue
        for key in snapshot.keys():
            value = snapshot[key]
            if key in seen_values:
                other_stage, other_value = seen_values[key]
                if other_value != value:
                    raise ConfigError(
                        f"conflicting saved configs: stage {spec.name!r} "
                        f"and stage {other_stage!r} disagree on {key!r} "
                        f"({value!r} vs {other_value!r}); refusing to mix "
                        "them into one lineage"
                    )
            else:
                seen_values[key] = (spec.name, value)

    per_stage: Dict[str, DictConfig] = {}
    for spec in in_scope:
        base: dict = {}
        for prior in in_scope:
            if prior.name == spec.name:
                break
            prior_cfg = per_stage.get(prior.name)
            if prior_cfg is not None:
                base.update(OmegaConf.to_container(prior_cfg, resolve=False))

        explicit = _own_config_for(spec, configs_by_role)
        if explicit is not None:
            base.update(OmegaConf.to_container(explicit, resolve=False))
        else:
            own_snapshot = loaded.get(spec.name)
            if own_snapshot is not None:
                base.update(OmegaConf.to_container(own_snapshot, resolve=False))

        per_stage[spec.name] = OmegaConf.create(base)

    result = per_stage[requested_stage]
    for key, value in cli_dir_overrides.items():
        OmegaConf.update(result, key, value, merge=False)

    resolve_loaded_configs({requested_stage: result})
    return result
