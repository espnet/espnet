"""Checking a system's declared stage order against its own methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    """One stage of a system: its method name, config role, and log path.

    Args:
        name: The stage method name on the system (e.g. ``"train"``).
        config: The config role the stage runs on - any non-empty
            identifier, naming an ``*_config`` constructor argument the
            system takes (``"training"`` -> ``training_config``). A
            system's full set of roles is derived from its ``stages``
            (see :func:`roles`); there is no separate, fixed role list.
        log_dir: A dotted key inside that config whose value is the
            stage's log directory (e.g. ``"exp_dir"``, ``"pack.out_dir"``);
            ``None`` falls back to the system's own default log directory.

    Examples:
        >>> StageSpec(name="train", config="training", log_dir="exp_dir")
        StageSpec(name='train', config='training', log_dir='exp_dir')
    """

    name: str
    config: str
    log_dir: str | None = None


class StageContractError(ValueError):
    """A system's declared stages, or a stage request, are invalid.

    Raised when a system's declared stages do not match its own methods,
    or when a requested stage's config role was not given.

    Examples:
        >>> raise StageContractError("ASRSystem has no stage 'decode'")
        Traceback (most recent call last):
        espnet3.components.contract.stages.StageContractError: ASRSystem ...
    """


def check_stage_contract(cls: type) -> None:
    """Raise unless ``cls`` declares a valid, complete ``stages`` tuple.

    Checked once, when ``cls`` is defined (``BaseSystem.__init_subclass__``):
    ``stages`` is a non-empty tuple of :class:`StageSpec` with distinct
    names, each naming a non-empty identifier ``config`` role and a
    callable method of that name on ``cls``. Every public callable ``cls``
    defines (not starting with ``_``) must also be one of the declared
    stage names - a system has no other public surface.

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
        espnet3.components.contract.stages.StageContractError: BadSystem.stages must ...
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
        if not spec.config or not spec.config.isidentifier():
            raise StageContractError(
                f"{cls.__qualname__} stage {spec.name!r}: config must be a "
                f"non-empty identifier, not {spec.config!r}"
            )
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


def stage_names(system_cls: type) -> list[str]:
    """Return ``system_cls``'s stage names, in declared order.

    Examples:
        >>> class ExampleSystem:
        ...     stages = (StageSpec(name="train", config="training"),)
        >>> stage_names(ExampleSystem)
        ['train']
    """
    return [s.name for s in system_cls.stages]


def roles(system_cls: type) -> tuple[str, ...]:
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
        >>> roles(ExampleSystem)
        ('training', 'inference')
    """
    seen: dict[str, None] = {}
    for spec in system_cls.stages:
        seen.setdefault(spec.config, None)
    return tuple(seen)


def stage_spec(system_cls: type, name: str) -> StageSpec:
    """Return the declared :class:`StageSpec` named ``name`` on ``system_cls``.

    Raises:
        StageContractError: No stage named ``name`` is declared.

    Examples:
        >>> class ExampleSystem:
        ...     stages = (
        ...         StageSpec(name="train", config="training", log_dir="exp_dir"),
        ...     )
        >>> stage_spec(ExampleSystem, "train").config
        'training'
    """
    for spec in system_cls.stages:
        if spec.name == name:
            return spec
    raise StageContractError(
        f"{system_cls.__name__} has no stage {name!r}; declared stages are "
        f"{stage_names(system_cls)}"
    )


def stage_log_dir(system: Any, name: str) -> Path:
    """Return the log directory for stage ``name`` on ``system``.

    Reads the stage's declared ``config`` role's config object off
    ``system`` (e.g. ``system.training_config``) and its ``log_dir``
    dotted key, falling back to ``system._default_log_dir`` when the
    config, the key, or its value is absent.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> class ExampleSystem:
        ...     stages = (
        ...         StageSpec(name="train", config="training", log_dir="exp_dir"),
        ...     )
        ...     training_config = OmegaConf.create({"exp_dir": "exp"})
        ...     _default_log_dir = Path("logs")
        >>> stage_log_dir(ExampleSystem(), "train")
        PosixPath('exp')
    """
    from omegaconf import OmegaConf

    spec = stage_spec(type(system), name)
    config = getattr(system, f"{spec.config}_config", None)
    value = (
        OmegaConf.select(config, spec.log_dir)
        if (config is not None and spec.log_dir)
        else None
    )
    return Path(value) if value else system._default_log_dir


def check_requested_stages(
    system_cls: type, requested: Sequence[str], provided: Mapping[str, Any]
) -> None:
    """Raise unless each requested stage is declared and its config is given.

    Checked once, right after ``resolve_stages`` expands the requested
    stage list, before any config is loaded.

    Args:
        system_cls: The system class the stages are requested against.
        requested: The resolved stage names (``resolve_stages``'s output).
        provided: Config role -> the config passed for that role (or
            ``None``/absent when not given), e.g. ``{"training": cfg,
            "inference": None}``.

    Raises:
        StageContractError: A requested name is not declared, or a
            declared stage's config role was not given.

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
    missing = [
        (s, names[s].config) for s in requested if provided.get(names[s].config) is None
    ]
    if missing:
        lines = "\n".join(f"  - {s} runs on the {role} config" for s, role in missing)
        raise StageContractError(
            f"config not provided for requested stage(s):\n{lines}"
        )


__all__ = [
    "StageContractError",
    "StageSpec",
    "check_requested_stages",
    "check_stage_contract",
    "roles",
    "stage_log_dir",
    "stage_names",
    "stage_spec",
]
