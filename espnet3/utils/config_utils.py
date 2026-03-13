"""Configuration helpers and OmegaConf resolvers for ESPnet3."""

import logging
import re
from importlib import resources
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

_RELATIVE_RESOLVER_PATTERN = re.compile(
    r"\$\{(?P<resolver>load_line|load_yaml):(?P<path>[^,}]+)"
)


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


def load_yaml(path, key=None):
    """Load a YAML file and optionally return a nested key.

    This resolver is intended for pulling a single value from another config,
    e.g., `${load_yaml:conf/train.yaml,exp_tag}`.

    Args:
        path (str or Path): Path to the YAML file.
        key (str, optional): Dot-delimited key to select.

    Returns:
        Any: The selected value or the full config if key is None.
    """
    cfg_path = Path(path)
    try:
        cfg = OmegaConf.load(cfg_path)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except PermissionError:
        logging.error(f"Permission denied when accessing file: {path}")
        raise

    if key is None or str(key).strip() == "":
        return cfg

    value = OmegaConf.select(cfg, str(key), default=None)
    if value is None:
        raise KeyError(f"Key not found in YAML: {key}")
    return value


OMEGACONF_ESPNET3_RESOLVER = {
    "load_line": load_line,
    "load_yaml": load_yaml,
}
for name, resolver in OMEGACONF_ESPNET3_RESOLVER.items():
    OmegaConf.register_new_resolver(name, resolver)
    logging.info(f"Registered ESPnet-3 OmegaConf Resolver: {name}")


def _process_dict_config_entry(
    entry: DictConfig,
    base_path: Path,
    resolve: bool = True,
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
            {key: load_config_with_defaults(str(config_path), resolve=resolve)}
        )
    return results


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
    base_path = Path(path).parent
    main_config = OmegaConf.load(path)
    _normalize_relative_resolver_paths(main_config, base_path)
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
                    load_config_with_defaults(str(config_path), resolve=resolve)
                )

        elif isinstance(entry, DictConfig):
            merged_configs.extend(
                _process_dict_config_entry(entry, base_path, resolve=resolve)
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


def load_template_defaults(
    template_config_path: str,
    template_package: str,
):
    """Load packaged TEMPLATE defaults by config kind."""
    resource = resources.files(template_package).joinpath(
        *Path(template_config_path).parts
    )
    with resources.as_file(resource) as path:
        return load_config_with_defaults(str(path), resolve=False)


def load_and_merge_config(
    config_path: Path | None,
    template_config_path: str,
    template_package: str | None = None,
):
    """Load user config and merge it with TEMPLATE defaults."""
    if config_path is None:
        return None
    if template_package is None:
        template_package = _infer_template_package_from_config_path(config_path)
    if template_package is None:
        raise ValueError(
            "template_package is required when it cannot be inferred from config_path"
        )
    default_cfg = load_template_defaults(template_config_path, template_package)
    user_cfg = load_config_with_defaults(str(config_path), resolve=False)
    merged_cfg = OmegaConf.merge(default_cfg, user_cfg)
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


def _normalize_relative_resolver_paths(cfg, base_path: Path) -> None:
    if isinstance(cfg, DictConfig):
        for key in list(cfg.keys()):
            value = cfg._get_node(key)
            if isinstance(value, (DictConfig, ListConfig)):
                _normalize_relative_resolver_paths(value, base_path)
            else:
                raw_value = value._value()
                if isinstance(raw_value, str):
                    cfg[key] = _rewrite_relative_resolver_paths(raw_value, base_path)
    elif isinstance(cfg, ListConfig):
        for idx in range(len(cfg)):
            value = cfg._get_node(idx)
            if isinstance(value, (DictConfig, ListConfig)):
                _normalize_relative_resolver_paths(value, base_path)
            else:
                raw_value = value._value()
                if isinstance(raw_value, str):
                    cfg[idx] = _rewrite_relative_resolver_paths(raw_value, base_path)


def _rewrite_relative_resolver_paths(value: str, base_path: Path) -> str:
    def replace(match: re.Match) -> str:
        resolver = match.group("resolver")
        raw_path = match.group("path").strip()

        quote = ""
        normalized_path = raw_path
        if (
            len(raw_path) >= 2
            and raw_path[0] == raw_path[-1]
            and raw_path[0]
            in (
                "'",
                '"',
            )
        ):
            quote = raw_path[0]
            normalized_path = raw_path[1:-1].strip()

        # Leave dynamic or already-absolute paths untouched.
        if "${" in normalized_path or Path(normalized_path).is_absolute():
            return match.group(0)

        base_relative_path = base_path / normalized_path
        workspace_relative_path = Path(normalized_path)
        if normalized_path.startswith(("./", "../")) or base_relative_path.exists():
            resolved_path = base_relative_path.absolute().as_posix()
        elif workspace_relative_path.exists():
            resolved_path = workspace_relative_path.absolute().as_posix()
        else:
            resolved_path = base_relative_path.absolute().as_posix()
        if quote:
            resolved_path = f"{quote}{resolved_path}{quote}"
        return f"${{{resolver}:{resolved_path}"

    return _RELATIVE_RESOLVER_PATTERN.sub(replace, value)


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


def _infer_template_package_from_config_path(config_path: Path) -> str | None:
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
