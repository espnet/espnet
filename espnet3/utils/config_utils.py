"""Configuration helpers and OmegaConf resolvers for ESPnet3."""

import logging
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


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


OMEGACONF_ESPNET3_RESOLVER = {
    "load_line": load_line,
}
for name, resolver in OMEGACONF_ESPNET3_RESOLVER.items():
    OmegaConf.register_new_resolver(name, resolver)
    logging.info(f"Registered ESPnet-3 OmegaConf Resolver: {name}")


def _process_dict_config_entry(entry: DictConfig, base_path: Path) -> list:
    results = []
    for key, val in entry.items():
        # Skip null entries (can be used for disabling config entries)
        if val is None:
            continue

        # Compose config path like 'optim/adam.yaml'
        # or use as-is if path includes '/'
        composed = f"{key}/{val}" if "/" not in val else val
        config_path = _build_config_path(base_path, composed)
        results.append({key: load_config_with_defaults(str(config_path))})
    return results


def load_config_with_defaults(path: str) -> OmegaConf:
    """Load an YAML config with recursive `_self_` merging via Hydra-style `defaults`.

    This function recursively loads and merges dependent YAML files specified
    in the `defaults` key of the given config. It mimics Hydra’s composition mechanism
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

    Returns:
        OmegaConf.DictConfig: Fully resolved and merged configuration object.
    """
    base_path = Path(path).parent
    main_config = OmegaConf.load(path)
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
                merged_configs.append(load_config_with_defaults(str(config_path)))

        elif isinstance(entry, DictConfig):
            merged_configs.extend(_process_dict_config_entry(entry, base_path))

        elif entry == "_self_":
            merged_configs.append(config_self)
            self_merged = True

    if not self_merged:
        merged_configs.append(config_self)

    final_config = OmegaConf.merge(*merged_configs)
    OmegaConf.resolve(final_config)

    if "defaults" in final_config:
        del final_config["defaults"]

    return final_config


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
