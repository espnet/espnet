"""Mini AN4 SLU dataset: an intent label prefixed to an AN4 transcript."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.mini_an4.esp2_slu.dataset.builder import MiniAn4SluBuilder, resolve_data_root
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


class MiniAn4SluDataset(BaseDataset):
    """AN4 served the way ``egs3/slurp/esp2_slu`` serves SLURP.

    The speaker id stands in for SLURP's ``scenario_action`` label, so ``text``
    is ``"<intent> <transcript>"`` and the same system, runner and metrics run
    over it. That is what makes this a usable integration fixture: it exercises
    the SLU path without SLURP's 58 GB of audio or a BERT post-decoder.

    Args:
        split: ``train``, ``valid`` or ``test``.
        recipe_dir: Recipe directory; defaults to this module's recipe.
        with_references: Whether to also return ``intent`` and
            ``reference_transcript``, the columns ``metrics.yaml`` binds with
            ``dataset:``. They are plain strings, which the training collate
            function cannot pad, so inference turns them on and training
            leaves them off.

    Raises:
        ValueError: If ``split`` is unknown.

    Examples:
        >>> dataset = MiniAn4SluDataset(split="test")  # doctest: +SKIP
        >>> sorted(dataset[0])  # doctest: +SKIP
        ['speech', 'text']
    """

    fields = (
        Field("speech", "audio"),
        Field("text", "text"),
        Field("transcript", "text", optional=True),
        Field("intent", "text", optional=True),
        Field("reference_transcript", "text", optional=True),
    )

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        with_references: bool = False,
    ) -> None:
        self.split = str(split)
        self.with_references = bool(with_references)
        if self.split not in _SPLIT_MANIFEST_PATHS:
            known = ", ".join(sorted(_SPLIT_MANIFEST_PATHS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        builder = MiniAn4SluBuilder()
        if not builder.is_built(recipe_dir=recipe_root):
            builder.build(recipe_dir=recipe_root)

        manifest = resolve_data_root(recipe_root) / _SPLIT_MANIFEST_PATHS[self.split]
        self._rows = [
            line.split("\t", maxsplit=3)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        _utt_id, wav_path, intent, transcript = self._rows[int(idx)]
        array, _rate = sf.read(wav_path)
        sample = {
            "speech": np.asarray(array, dtype=np.float32),
            "text": f"{intent} {transcript}",
        }
        if self.with_references:
            sample["intent"] = intent
            sample["reference_transcript"] = transcript
        return sample
