import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_line(path):
    """
    Load lines from a text file and return as a list of strings.

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
    """
    Process a dictionary-style entry in the `defaults` list and return a list
    of partial configs to be merged.
    """
    results = []
    for key, val in entry.items():
        # Skip null entries (can be used for disabling config entries)
        if val is None:
            continue

        # Compose config path like 'optimizer/adam.yaml'
        # or use as-is if path includes '/'
        composed = f"{key}/{val}" if "/" not in val else val
        cfg_path = _build_config_path(base_path, composed)
        results.append({key: load_config_with_defaults(str(cfg_path))})
    return results


def load_config_with_defaults(path: str) -> OmegaConf:
    """
    Load an OmegaConf YAML config file with support for recursive `_self_` merging
    based on Hydra-style `defaults` lists.

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
          - optimizer: adam
          - _self_

        # This will recursively load:
        #   model/conformer.yaml
        #   optimizer/adam.yaml
        # and merge them with config.yaml itself at the end.

    Args:
        path (str): Path to the main YAML config file.

    Returns:
        OmegaConf.DictConfig: Fully resolved and merged configuration object.
    """
    base_path = Path(path).parent
    main_cfg = OmegaConf.load(path)
    cfg_self = main_cfg.copy()

    if "defaults" not in main_cfg:
        return cfg_self

    merged_cfgs = []
    self_merged = False

    for entry in main_cfg.defaults:
        if isinstance(entry, str):
            if entry == "_self_":
                merged_cfgs.append(cfg_self)
                self_merged = True
            else:
                cfg_path = _build_config_path(base_path, entry)
                merged_cfgs.append(load_config_with_defaults(str(cfg_path)))

        elif isinstance(entry, DictConfig):
            merged_cfgs.extend(_process_dict_config_entry(entry, base_path))

        elif entry == "_self_":
            merged_cfgs.append(cfg_self)
            self_merged = True

    if not self_merged:
        merged_cfgs.append(cfg_self)

    final_cfg = OmegaConf.merge(*merged_cfgs)
    OmegaConf.resolve(final_cfg)

    final_cfg = _normalize_dataset_config(final_cfg, Path(path))

    if "defaults" in final_cfg:
        del final_cfg["defaults"]

    return final_cfg


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


def _load_module_from_path(path: Path) -> ModuleType:
    """
    Load a Python module from a specific file path without mutating sys.path.

    This helper is used to import recipe-local modules such as ``src/dataset.py``
    even when that directory is not on PYTHONPATH. It assigns a temporary, unique
    module name to avoid collisions and returns the loaded module object.
    """
    module_name = f"_espnet3_dynamic_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def _discover_dataset_class(module: ModuleType) -> str:
    """
    Return the sole Dataset-like class name defined in the module.

    A class qualifies if it:
    - is defined in the module itself (not imported), and
    - has a name ending with ``Dataset``.

    Raises if zero or multiple candidates are found to force explicit `_target_`.
    """
    candidates = [
        name
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and name.endswith("Dataset")
    ]

    if len(candidates) == 0:
        raise ValueError(
            "No dataset class found in dataset module. "
            "Please specify dataset.dataset._target_ explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple dataset classes found in dataset module: {candidates}. "
            "Please specify dataset.dataset._target_ explicitly."
        )
    return candidates[0]


def _infer_dataset_module(config_dir: Path, dataset_cfg: DictConfig) -> tuple[str, str]:
    """
    Resolve the dataset module path and class name.

    - Uses `dataset_cfg.dataset_module` if provided.
    - Otherwise defaults to `<recipe_root>/src/dataset.py` and sets
      `dataset_cfg.dataset_module` accordingly.
    - Loads the module (from path if it exists, else via import), and discovers
      the single `*Dataset` class within it.

    Returns `(module_import_path, class_name)`.
    """
    dataset_module = dataset_cfg.get("dataset_module")
    if dataset_module is None:
        dataset_module = str(config_dir.parent / "src" / "dataset.py")
        dataset_cfg["dataset_module"] = dataset_module

    module_path = Path(str(dataset_module))
    if module_path.exists():
        module = _load_module_from_path(module_path)
        # Use canonical import path for readability when available
        module_name = (
            "src.dataset"
            if module_path.name == "dataset.py"
            else module.__name__
        )
    else:
        module = importlib.import_module(str(dataset_module))
        module_name = module.__name__

    class_name = _discover_dataset_class(module)
    return module_name, class_name


def _normalize_dataset_config(cfg: OmegaConf, config_path: Path) -> OmegaConf:
    """
    Normalize dataset configuration:

    - Default `dataset._target_` to DataOrganizer if missing.
    - Default `dataset.dataset_module` to `<recipe>/src/dataset.py` if missing.
    - Auto-fill `_target_` for train/valid/test dataset entries when omitted by
      discovering the single `*Dataset` class in the resolved dataset module.
    """
    if "dataset" not in cfg:
        return cfg

    dataset_cfg = cfg.dataset
    if not isinstance(dataset_cfg, DictConfig):
        return cfg

    if "_target_" not in dataset_cfg:
        dataset_cfg["_target_"] = "espnet3.components.data_organizer.DataOrganizer"

    config_dir = config_path.parent
    module_name = class_name = None

    def ensure_targets(split_key: str):
        nonlocal module_name, class_name
        split = dataset_cfg.get(split_key)
        if not split:
            return
        for entry in split:
            if not entry or "dataset" not in entry:
                continue
            ds_cfg = entry["dataset"]
            if isinstance(ds_cfg, DictConfig) and "_target_" not in ds_cfg:
                if module_name is None or class_name is None:
                    module_name, class_name = _infer_dataset_module(
                        config_dir, dataset_cfg
                    )
                ds_cfg["_target_"] = f"{module_name}.{class_name}"

    for key in ("train", "valid", "test"):
        ensure_targets(key)

    return cfg
