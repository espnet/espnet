"""Make a packed bundle's ESPnet2 train config load on any machine."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

logger = logging.getLogger(__name__)


def inline_token_list(bundle_dir: Path) -> None:
    """Write the token list of the bundle's train config out as a list.

    The train stage saves ``token_list`` as the path of a file in the corpus,
    which exists only where the corpus does. espnet2 reads the config with
    ``yaml.safe_load`` and opens that path as it stands, so a bundle copied
    elsewhere cannot build its model. espnet2's own training saves the list
    itself, "to keep it as portable" (``S2TTask.build_model``); this writes
    it the same way, read from the file as espnet2 reads it.

    The config rewritten is the one the bundled inference config decodes
    with. One that already holds a list is left as it is.

    Args:
        bundle_dir: The directory ``pack_model`` wrote.

    Raises:
        FileNotFoundError: If the token list the config names is missing.
    """
    bundle_dir = Path(bundle_dir).resolve()
    inference = OmegaConf.load(bundle_dir / "conf" / "inference.yaml")
    inference.recipe_dir = str(bundle_dir)
    path = OmegaConf.select(inference, "model.s2t_train_config")
    if path is None:
        return
    train_config = Path(path)
    if not train_config.is_absolute():
        train_config = bundle_dir / train_config
    train_config = train_config.resolve()
    if bundle_dir not in train_config.parents or not train_config.is_file():
        logger.warning(
            "The packed inference config decodes with %s, which is not in the "
            "bundle, so the bundle decodes only where that file exists.",
            train_config,
        )
        return
    args = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    if not isinstance(args.get("token_list"), str):
        return
    with open(args["token_list"], encoding="utf-8") as f:
        args["token_list"] = [line.rstrip() for line in f]
    with train_config.open("w", encoding="utf-8") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
