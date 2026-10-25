"""Checking a system's declared stage order against its own methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    """One stage of a system: its method name, config role(s), and log path.

    Args:
        name: The stage method name on the system (e.g. ``"train"``).
        config: The config role(s) the stage runs on - a non-empty
            identifier, or a tuple of them when the stage needs more than
            one, each naming an ``*_config`` constructor argument the
            system takes (``"training"`` -> ``training_config``). A
            system's full set of roles is derived from its ``stages``
            (see :func:`list_roles`); there is no separate, fixed role list.
        log_dir: A dotted key inside the first provided role's config
            whose value is the stage's log directory (e.g. ``"exp_dir"``,
            ``"pack.out_dir"``); ``None`` falls back to the system's own
            default log directory.

    Examples:
        >>> StageSpec(name="train", config="training", log_dir="exp_dir")
        StageSpec(name='train', config='training', log_dir='exp_dir')
        >>> StageSpec(name="pack_model", config=("training", "publication"))
        StageSpec(name='pack_model', config=('training', 'publication'), log_dir=None)
    """

    name: str
    config: str | tuple[str, ...]
    log_dir: str | None = None
    dir_aliases: tuple[str, ...] = ()


def resolve_config_roles(config: str | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a :attr:`StageSpec.config` value to a tuple of roles.

    A bare string is treated as a single-role tuple; a tuple is returned
    as-is. Does not validate the roles themselves - see
    :func:`_validate_config_roles` for that.

    Examples:
        >>> resolve_config_roles("training")
        ('training',)
        >>> resolve_config_roles(("training", "publication"))
        ('training', 'publication')
    """
    return (config,) if isinstance(config, str) else tuple(config)


def _validate_config_roles(
    cls_name: str, stage_name: str, config: Any
) -> tuple[str, ...]:
    """Return ``config`` as a validated tuple of role identifiers.

    Raises:
        StageContractError: ``config`` is not a ``str`` or a ``tuple`` of
            ``str``, declares no role, repeats a role, or a role is not a
            non-empty identifier.
    """
    if isinstance(config, str):
        config_roles_: tuple[str, ...] = (config,)
    elif isinstance(config, tuple) and all(isinstance(c, str) for c in config):
        config_roles_ = config
    else:
        raise StageContractError(
            f"{cls_name} stage {stage_name!r}: config must be a str or "
            f"tuple[str, ...], not {config!r}"
        )
    if not config_roles_:
        raise StageContractError(
            f"{cls_name} stage {stage_name!r}: config must declare at least " "one role"
        )
    if len(set(config_roles_)) != len(config_roles_):
        raise StageContractError(
            f"{cls_name} stage {stage_name!r}: config declares duplicate "
            f"role(s): {config!r}"
        )
    for role in config_roles_:
        if not role.isidentifier():
            raise StageContractError(
                f"{cls_name} stage {stage_name!r}: config role must be a "
                f"non-empty identifier, not {role!r}"
            )
    return config_roles_


class StageContractError(ValueError):
    """A system's declared stages, or a stage request, are invalid.

    Raised when a system's declared stages do not match its own methods,
    or when a requested stage's config role was not given.

    Examples:
        >>> raise StageContractError("ASRSystem has no stage 'decode'")
        Traceback (most recent call last):
        espnet3.contract.stages.StageContractError: ASRSystem ...
    """


def check_stage_contract(cls: type) -> None:
    """Raise unless ``cls`` declares a valid, complete ``stages`` tuple.

    Checked once, when ``cls`` is defined (``BaseSystem.__init_subclass__``):
    ``stages`` is a non-empty tuple of :class:`StageSpec` with distinct
    names, each naming one or more non-empty identifier ``config`` role(s)
    and a callable method of that name on ``cls``. Every public callable
    ``cls`` defines (not starting with ``_``) must also be one of the
    declared stage names - a system has no other public surface.

    The "public method" scan intentionally looks only at ``vars(cls)``
    (``cls.__dict__``), not the full MRO: it checks what ``cls`` itself
    newly defines or overrides, not methods it merely inherits unchanged.
    This is deliberate, not an oversight - each class in the hierarchy is
    checked once, when it is itself defined (``__init_subclass__`` checks
    every subclass automatically; ``BaseSystem`` has no superclass to
    trigger that hook, so its own module calls this function on itself
    explicitly), against the ``stages`` tuple in effect *at that point*.
    A method already validated on a parent does not need
    re-validating on every child that leaves it untouched. The one case
    this does not catch: a subclass that *narrows* ``stages`` (drops a
    name a parent declared) without also removing or privatizing the
    inherited method of that name - the now-undeclared method stays
    inherited, not locally defined, so this scan does not see it. No
    system in this codebase does that today; every subclass only adds to
    or overrides ``BaseSystem.stages``, never narrows it.

    Args:
        cls: The system class being defined.

    Raises:
        StageContractError: ``stages`` is malformed, a declared stage has
            no matching method, or a public method is not declared as a
            stage.

    Examples:
        >>> class BadSystem:
        ...     stages = ()
        >>> check_stage_contract(BadSystem)
        Traceback (most recent call last):
        espnet3.contract.stages.StageContractError: BadSystem.stages must ...
    """
    stages = getattr(cls, "stages", None)
    if (
        not isinstance(stages, tuple)
        or not stages
        or not all(isinstance(s, StageSpec) for s in stages)
    ):
        raise StageContractError(
            f"{cls.__qualname__}.stages must be a non-empty tuple of StageSpec"
        )
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise StageContractError(f"{cls.__qualname__}.stages repeats a name: {names}")
    for spec in stages:
        _validate_config_roles(cls.__qualname__, spec.name, spec.config)
        if spec.log_dir is not None and not isinstance(spec.log_dir, str):
            raise StageContractError(
                f"{cls.__qualname__} stage {spec.name!r}: log_dir must be a "
                "str or None"
            )
        if not callable(getattr(cls, spec.name, None)):
            raise StageContractError(
                f"{cls.__qualname__} declares stage {spec.name!r} but defines "
                "no method of that name"
            )

    declared = set(names)
    for attr_name, attr in vars(cls).items():
        if attr_name.startswith("_") or not callable(attr):
            continue
        if attr_name not in declared:
            raise StageContractError(
                f"{cls.__qualname__}.{attr_name} is public but not a stage; "
                f"make it a stage or rename it _{attr_name}"
            )


def list_stage_names(system_cls: type) -> list[str]:
    """Return ``system_cls``'s stage names, in declared order.

    Examples:
        >>> class ExampleSystem:
        ...     stages = (StageSpec(name="train", config="training"),)
        >>> list_stage_names(ExampleSystem)
        ['train']
    """
    return [s.name for s in system_cls.stages]


def list_roles(system_cls: type) -> tuple[str, ...]:
    """Return ``system_cls``'s config roles, in stage order, without duplicates.

    Derived from each declared stage's :attr:`StageSpec.config` - there is
    no separate, fixed role list to keep in sync with what a system
    actually declares.

    Examples:
        >>> class ExampleSystem:
        ...     stages = (
        ...         StageSpec(name="train", config="training"),
        ...         StageSpec(name="infer", config="inference"),
        ...         StageSpec(name="measure", config="inference"),
        ...     )
        >>> list_roles(ExampleSystem)
        ('training', 'inference')
    """
    seen: dict[str, None] = {}
    for spec in system_cls.stages:
        for role in resolve_config_roles(spec.config):
            seen.setdefault(role, None)
    return tuple(seen)


def find_stage_spec(system_cls: type, name: str) -> StageSpec:
    """Return the declared :class:`StageSpec` named ``name`` on ``system_cls``.

    Raises:
        StageContractError: No stage named ``name`` is declared.

    Examples:
        >>> class ExampleSystem:
        ...     stages = (
        ...         StageSpec(name="train", config="training", log_dir="exp_dir"),
        ...     )
        >>> find_stage_spec(ExampleSystem, "train").config
        'training'
    """
    for spec in system_cls.stages:
        if spec.name == name:
            return spec
    raise StageContractError(
        f"{system_cls.__name__} has no stage {name!r}; declared stages are "
        f"{list_stage_names(system_cls)}"
    )


def resolve_stage_log_dir(config: Any, spec: StageSpec) -> Path | None:
    """Return a stage's log directory from its own merged config, if any.

    A pure function: reads ``spec.log_dir`` (a dotted key, e.g.
    ``"exp_dir"``, ``"pack.out_dir"``) off ``config``. Returns ``None``
    when ``spec.log_dir`` is unset or the key has no value in ``config``,
    leaving the fallback (the system's experiment directory, then a
    generic default) to the caller. ``config`` is the stage's own,
    already-merged config - when a stage declares more than one role
    (:attr:`StageSpec.config` as a tuple), the caller is responsible for
    merging those roles' configs into one before calling this (see
    ``espnet3.utils.stage_configs``); this function does not pick among
    roles itself.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> spec = StageSpec(name="train", config="training", log_dir="exp_dir")
        >>> resolve_stage_log_dir(OmegaConf.create({"exp_dir": "exp"}), spec)
        PosixPath('exp')
        >>> resolve_stage_log_dir(OmegaConf.create({}), spec) is None
        True
    """
    from omegaconf import OmegaConf

    if config is None or not spec.log_dir:
        return None
    value = OmegaConf.select(config, spec.log_dir)
    return Path(value) if value else None


def check_requested_stages(
    system_cls: type,
    requested: Sequence[str],
    provided: Mapping[str, Any],
    *,
    exp_dir: Any | None = None,
) -> None:
    """Raise unless each requested stage is declared and can get a config.

    Checked once, right after ``resolve_stages`` expands the requested
    stage list, before any config is loaded.

    Args:
        system_cls: The system class the stages are requested against.
        requested: The resolved stage names (``resolve_stages``'s output).
        provided: Config role -> the config passed for that role (or
            ``None``/absent when not given), e.g. ``{"training": cfg,
            "inference": None}``.
        exp_dir: The experiment directory, when given. A stage whose role
            config is missing is not an error when ``exp_dir`` is given:
            it may inherit that role from an earlier run's baked config
            instead; a config missing there too surfaces later, from
            config inheritance, not from this check.

    Raises:
        StageContractError: A requested name is not declared, or (only
            when ``exp_dir`` is not given) a declared stage's config role
            was not given either.

    Examples:
        >>> class ExampleSystem:
        ...     stages = (StageSpec(name="train", config="training"),)
        >>> check_requested_stages(ExampleSystem, ["train"], {"training": object()})
    """
    names = {s.name: s for s in system_cls.stages}
    unknown = [s for s in requested if s not in names]
    if unknown:
        raise StageContractError(
            f"{system_cls.__name__} has no stage {unknown}; declared stages "
            f"are {list(names)}"
        )
    if exp_dir is not None:
        return
    missing = [
        (s, role)
        for s in requested
        for role in resolve_config_roles(names[s].config)
        if provided.get(role) is None
    ]
    if missing:
        lines = "\n".join(f"  - {s} runs on the {role} config" for s, role in missing)
        raise StageContractError(
            f"config not provided for requested stage(s) (pass --exp_dir to "
            f"inherit from an earlier run instead):\n{lines}"
        )


def generate_stage_flag_mapping(
    stage_specs,
) -> dict[str, str]:
    """Return a mapping from CLI flag names to their canonical config key.

    For every stage with a valid ``log_dir``, computes the CLI flag names
    (the canonical ``log_dir`` key itself, plus any ``dir_aliases``) and
    maps each one to the single canonical config key it writes to. Multiple
    stages may declare the same canonical key; their flag names are simply
    merged, since they all write to the same place.

    Args:
        stage_specs: All declared stages of one or more systems.

    Returns:
        A dict mapping each unique CLI flag name to the single canonical
        config key (``StageSpec.log_dir``) it writes to. A canonical key and
        its aliases all map to the SAME canonical key string - callers must
        use this value (not the flag name) as the config key/argparse
        destination, so an alias and its canonical name always write to the
        same place.

    Raises:
        StageContractError: A flag name would be derived from two different
            canonical keys.
    """
    flag_name_to_canonical: dict[str, str] = {}

    for spec in stage_specs:
        if spec.log_dir is None:
            continue

        canonical_key = spec.log_dir
        location_keys = list(spec.dir_aliases) + [canonical_key]

        for key in location_keys:
            flag_name = key.replace(".", "_")

            if flag_name in flag_name_to_canonical:
                existing_canonical = flag_name_to_canonical[flag_name]
                if existing_canonical != canonical_key:
                    raise StageContractError(
                        f"CLI flag name collision: flag {flag_name!r} "
                        "would be derived from "
                        f"both canonical key {canonical_key!r} "
                        f"(via StageSpec name={spec.name!r}) "
                        f"and already-defined canonical key {existing_canonical!r}"
                    )
            else:
                flag_name_to_canonical[flag_name] = canonical_key

    return flag_name_to_canonical


__all__ = [
    "StageContractError",
    "StageSpec",
    "check_requested_stages",
    "check_stage_contract",
    "resolve_config_roles",
    "list_roles",
    "resolve_stage_log_dir",
    "list_stage_names",
    "generate_stage_flag_mapping",
]
