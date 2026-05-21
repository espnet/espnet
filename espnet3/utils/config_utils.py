"""Configuration helpers and OmegaConf resolvers for ESPnet3."""

import logging
import re
from importlib import resources
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

# Used to rewrite relative paths in resolver expressions such as
# `${load_line:conf/tokens.txt}` during config loading.
_RELATIVE_RESOLVER_PATTERN = re.compile(r"\$\{(?P<resolver>load_line):(?P<path>[^,}]+)")


def load_line(path):
    """Load lines from a text file and return as a list of strings.

    This function is used as a custom resolver in OmegaConf,
    allowing YAML files to reference external text line files
    dynamically via `${load_line:some/file.txt}`.

    This resolver is intended to load vocab file in configuration.

    Args:
        path (str or Path): Path to the file.

    Returns:
        List[str]: A list of stripped lines from the file.
    """
    try:
        with open(path, "r") as f:
            return ListConfig([line.strip() for line in f.readlines()])
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except PermissionError:
        logging.error(f"Permission denied when accessing file: {path}")
        raise


def self_name(path):
    """Return the current config stem for `${self_name:}` interpolation.

    This resolver supports config values such as `exp_tag: ${self_name:}`.
    During config loading, `load_config_with_defaults()` rewrites that
    interpolation to the stem of the YAML file currently being processed.

    The resolver itself is intentionally a passthrough because OmegaConf only
    passes explicit resolver arguments to callbacks. The actual config-name
    injection happens earlier in `_normalize_relative_resolver_paths()`, which
    replaces `${self_name:}` with the plain config stem before resolution.

    Args:
        path (str): Config stem passed explicitly to the resolver, such as
            `training` or `inference`.

    Returns:
        str: The injected config stem unchanged.

    Notes:
        This resolver is intended for ESPnet config composition. It is
        primarily consumed from YAML rather than called directly from Python.

    Examples:
        In a config file named `training.yaml`:

            exp_tag: ${self_name:}

        The expression is rewritten during loading to:

            training
    """
    return path


OMEGACONF_ESPNET3_RESOLVER = {
    "load_line": load_line,
    "self_name": self_name,
}
for name, resolver in OMEGACONF_ESPNET3_RESOLVER.items():
    OmegaConf.register_new_resolver(name, resolver)
    logging.info(f"Registered ESPnet-3 OmegaConf Resolver: {name}")


def _process_dict_config_entry(
    entry: DictConfig,
    base_path: Path,
    resolve: bool = True,
    bind_self_name: bool = True,
) -> list:
    results = []
    for key, val in entry.items():
        # Skip null entries (can be used for disabling config entries)
        if val is None:
            continue

        # Compose config path like 'optim/adam.yaml'
        # or use as-is if path includes '/'
        composed = f"{key}/{val}" if "/" not in val else val
        config_path = _build_config_path(base_path, composed)
        results.append(
            {
                key: _load_config_with_defaults(
                    str(config_path),
                    resolve=resolve,
                    bind_self_name=bind_self_name,
                )
            }
        )
    return results


def _load_config_with_defaults(
    path: str, resolve: bool = True, bind_self_name: bool = True
) -> OmegaConf:
    """Load a config while optionally deferring `${self_name:}` binding."""
    config_path = Path(path)
    base_path = config_path.parent
    main_config = OmegaConf.load(path)
    # OmegaConf resolvers only receive their explicit arguments, so resolver
    # callbacks cannot know which parent YAML file referenced them. Rewrite
    # relative resolver paths here while we still know the path of the file
    # currently being loaded.
    _normalize_relative_resolver_paths(
        main_config,
        base_path,
        config_path.stem if bind_self_name else None,
    )
    config_self = main_config.copy()

    if "defaults" not in main_config:
        return config_self

    merged_configs = []
    self_merged = False

    for entry in main_config.defaults:
        if isinstance(entry, str):
            if entry == "_self_":
                merged_configs.append(config_self)
                self_merged = True
            else:
                config_path = _build_config_path(base_path, entry)
                merged_configs.append(
                    _load_config_with_defaults(
                        str(config_path),
                        resolve=resolve,
                        bind_self_name=bind_self_name,
                    )
                )

        elif isinstance(entry, DictConfig):
            merged_configs.extend(
                _process_dict_config_entry(
                    entry,
                    base_path,
                    resolve=resolve,
                    bind_self_name=bind_self_name,
                )
            )

        elif entry == "_self_":
            merged_configs.append(config_self)
            self_merged = True

    if not self_merged:
        merged_configs.append(config_self)

    final_config = OmegaConf.merge(*merged_configs)

    if resolve:
        OmegaConf.resolve(final_config)

    if "defaults" in final_config:
        del final_config["defaults"]

    _ensure_target_convert_all(final_config)
    return final_config


def load_config_with_defaults(path: str, resolve: bool = True) -> OmegaConf:
    """Load an YAML config with recursive `_self_` merging via Hydra-style `defaults`.

    This function recursively loads and merges dependent YAML files specified
    in the `defaults` key of the given config. It mimics Hydra's composition mechanism
    without using Hydra's runtime, which makes it suitable for standalone YAML handling
    (e.g., for distributed training or script-based training setups).

    Supported formats inside `defaults`:
    - `"subconfig"` → loads `subconfig.yaml`
    - `{"key": "value"}` → loads `key/value.yaml`
    - `"_self_"` → appends the current config in-place

    Example:
        # config.yaml
        defaults:
          - model: conformer
          - optim: adam
          - _self_

        # This will recursively load:
        #   model/conformer.yaml
        #   optim/adam.yaml
        # and merge them with config.yaml itself at the end.

    Args:
        path (str): Path to the main YAML config file.
        resolve (bool): Whether to resolve interpolations after merging.

    Returns:
        OmegaConf.DictConfig: Fully resolved and merged configuration object.
    """
    return _load_config_with_defaults(path, resolve=resolve, bind_self_name=True)


def _load_default_config(
    config_name: str,
    default_package: str,
    bind_self_name: bool = True,
):
    """Load a packaged default config while optionally deferring self-name binding."""
    resource = resources.files(default_package).joinpath("conf", config_name)
    with resources.as_file(resource) as path:
        return _load_config_with_defaults(
            str(path), resolve=False, bind_self_name=bind_self_name
        )


def load_default_config(
    config_name: str,
    default_package: str,
):
    """Load a packaged default config without resolving interpolations.

    This helper reads a default config bundled in an ESPnet recipe package.
    The default package is typically an `egs3` recipe package such as
    `egs3.TEMPLATE.asr`, but any installed recipe package with a `conf/`
    directory can be used as the source of default values. The loaded config
    is intended to be merged later with a user-provided config via
    `load_and_merge_config()`.

    Example:
        If `default_package` is `egs3.TEMPLATE.asr` and
        `config_name` is `training.yaml`, this loads:

            egs3/TEMPLATE/asr/conf/training.yaml

        If you want to base a new recipe on an existing one, you can also
        point `default_package` to that recipe package. For example, using
        `egs3.librispeech.asr` with `training.yaml` would load:

            egs3/librispeech/asr/conf/training.yaml

    Args:
        config_name (str): Config filename under `conf/`, such as
            `training.yaml`, `inference.yaml`, or `metrics.yaml`.
        default_package (str): Python package that contains the default
            recipe resources. For example, `egs3.TEMPLATE.asr` points to files
            under `egs3/TEMPLATE/asr/`. Other installed recipe packages can
            also be used as long as they provide `conf/<config_name>`.

    Returns:
        OmegaConf.DictConfig: The default config loaded from package
        resources, with interpolation resolution deferred so it can be merged
        with user overrides first.
    """
    return _load_default_config(
        config_name,
        default_package,
        bind_self_name=True,
    )


def load_and_merge_config(
    config_path: Path | None,
    config_name: str,
    default_package: str | None = None,
    resolve: bool = True,
):
    """Load a user config and merge it with packaged default values.

    This is the higher-level helper used by recipe code. It first loads the
    default config through `load_default_config()`, then loads the user config
    without resolving interpolations, merges the two, and finally resolves the
    merged result. This allows user configs to reference values defined in the
    defaults, while also letting user-provided values override those defaults.

    The default source is usually an `egs3.TEMPLATE.*` package, but any
    installed recipe package can be used if it provides the same config file
    under its own `conf/` directory.

    Example:
        If a recipe config lives at:

            egs3/mini_an4/asr/conf/training.yaml

        and `config_name` is `training.yaml`, this function can infer
        `default_package="egs3.TEMPLATE.asr"` and merge:

            egs3/TEMPLATE/asr/conf/training.yaml

        with:

            egs3/mini_an4/asr/conf/training.yaml

    Args:
        config_path (Path | None): Path to the user config. If `None`, this
            function returns `None`.
        config_name (str): Config filename under `conf/`, such as
            `training.yaml`, `inference.yaml`, or `metrics.yaml`.
        default_package (str | None): Python package that contains the default
            recipe resources. If omitted, it is inferred from `config_path`.
            For example, a config under `egs3/<recipe>/asr/conf/` maps to
            `egs3.TEMPLATE.asr`.

    Returns:
        OmegaConf.DictConfig | None: The merged config. Interpolations are
        resolved after merging when ``resolve`` is ``True``. Returns ``None``
        if ``config_path`` is ``None``.
    """
    if config_path is None:
        return None
    if default_package is None:
        default_package = _infer_default_package_from_config_path(config_path)
    if default_package is None:
        raise ValueError(
            "default_package is required when it cannot be inferred from config_path"
        )
    default_cfg = _load_default_config(
        config_name,
        default_package,
        bind_self_name=False,
    )
    user_cfg = _load_config_with_defaults(
        str(config_path),
        resolve=False,
        bind_self_name=False,
    )
    merged_cfg = OmegaConf.merge(default_cfg, user_cfg)
    _normalize_relative_resolver_paths(merged_cfg, config_path.parent, config_path.stem)
    if resolve:
        OmegaConf.resolve(merged_cfg)
    _ensure_target_convert_all(merged_cfg)
    return merged_cfg


def _ensure_target_convert_all(cfg) -> None:
    if isinstance(cfg, DictConfig):
        if "_target_" in cfg:
            cfg["_convert_"] = "all"
        for value in cfg.values():
            _ensure_target_convert_all(value)
    elif isinstance(cfg, ListConfig):
        for value in cfg:
            _ensure_target_convert_all(value)


def _normalize_relative_resolver_paths(
    cfg, base_path: Path, config_name: str | None
) -> None:
    # This runs during config loading rather than inside load_yaml/load_line
    # because the resolver callback cannot access the path of the parent YAML.
    if isinstance(cfg, DictConfig):
        for key in list(cfg.keys()):
            value = cfg._get_node(key)
            if isinstance(value, (DictConfig, ListConfig)):
                _normalize_relative_resolver_paths(value, base_path, config_name)
            else:
                raw_value = value._value()
                if isinstance(raw_value, str):
                    if config_name is not None:
                        raw_value = raw_value.replace("${self_name:}", config_name)
                    cfg[key] = _rewrite_relative_resolver_paths(raw_value, base_path)
    elif isinstance(cfg, ListConfig):
        for idx in range(len(cfg)):
            value = cfg._get_node(idx)
            if isinstance(value, (DictConfig, ListConfig)):
                _normalize_relative_resolver_paths(value, base_path, config_name)
            else:
                raw_value = value._value()
                if isinstance(raw_value, str):
                    if config_name is not None:
                        raw_value = raw_value.replace("${self_name:}", config_name)
                    cfg[idx] = _rewrite_relative_resolver_paths(raw_value, base_path)


def _rewrite_relative_resolver_paths(value: str, base_path: Path) -> str:
    """Rewrite relative resolver paths to absolute paths for the current config.

    For example, if `value` contains `${load_line:tokens.txt}` and the current
    config lives under `conf/`, this rewrites the resolver path to point at
    `conf/tokens.txt` before OmegaConf resolves the expression.
    Absolute paths and dynamic paths are left unchanged.
    """

    def replace(match: re.Match) -> str:
        resolver = match.group("resolver")
        quote, normalized_path = _split_optional_quotes(match.group("path").strip())

        # Leave dynamic or already-absolute paths untouched.
        if "${" in normalized_path or Path(normalized_path).is_absolute():
            return match.group(0)

        resolved_path = (base_path / normalized_path).absolute().as_posix()
        # Keep the original quote style because absolute paths may contain
        # spaces on Windows or in user-managed workspaces.
        if quote:
            resolved_path = f"{quote}{resolved_path}{quote}"
        return f"${{{resolver}:{resolved_path}"

    return _RELATIVE_RESOLVER_PATTERN.sub(replace, value)


def _split_optional_quotes(raw_path: str) -> tuple[str, str]:
    # Resolver paths may be quoted when the path contains spaces. Preserve the
    # quote style so the rewritten resolver expression remains parseable.
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] and raw_path[0] in ("'", '"'):
        return raw_path[0], raw_path[1:-1].strip()
    return "", raw_path


def _build_config_path(base_path: Path, entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.suffix and entry_path.suffix != ".yaml":
        raise ValueError(
            f"Invalid file extension '{entry_path.suffix}' in entry: {entry}. "
            "Expected '.yaml'."
        )
    if not entry_path.suffix:
        entry += ".yaml"
    return base_path / entry


def _infer_default_package_from_config_path(config_path: Path) -> str | None:
    parts = config_path.resolve().parts
    try:
        egs3_index = parts.index("egs3")
    except ValueError:
        return None

    conf_index = None
    for index in range(egs3_index + 1, len(parts)):
        if parts[index] == "conf":
            conf_index = index
            break
    if conf_index is None or conf_index - egs3_index < 3:
        return None

    task_name = parts[conf_index - 1]
    return f"egs3.TEMPLATE.{task_name}"
