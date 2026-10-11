"""LibriSpeech dataset backed by Omni-IO archives and per-split views."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Any, List

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech.esp2_asr.dataset.builder import (
    LibriSpeechBuilder,
    blob_path,
    raw_archive_dir,
    read_text_table,
    split_view_path,
)
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}

_OMNIIO_HINT = (
    "Omni-IO is required by this recipe but is not installed. "
    "Install it with `pip install omniio` "
    "(it is not among ESPnet's default dependencies)."
)


def read_view(path: Path) -> dict[str, list]:
    """Read a split view table as plain Python lists.

    The row order is the one `create_dataset` wrote - each raw archive sorted
    by utterance id, archives in the order the split lists them - so the
    dataset index of an utterance does not depend on how the packing workers
    were scheduled.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    return {
        "id": list(table["id"]),
        "source": list(table["source"]),
        "bin_index": list(table["bin_index"]),
        "start_byte": list(table["start_byte"]),
        "end_byte": list(table["end_byte"]),
    }


class LibriSpeechDataset(TorchDataset):
    """Torch dataset that serves LibriSpeech out of Omni-IO archives.

    A split is a view over the per-raw-directory archives rather than a copy
    of them, so `dev` and `dev_clean`, or `train_100` and `train_960`, share
    the same blobs.

    Args:
        split: Logical split name such as ``train_960`` or ``test_clean``.
        recipe_dir: Optional recipe root. Defaults to this recipe directory.
        archive_root: Directory holding ``archive/``. Normally set to
            ``${data_dir}`` from the config.
        source_dir: Optional LibriSpeech parent/root override, used only when
            the archive still has to be built.
        build_if_missing: Pack the archive when it does not exist yet. Keep
            this ``False`` for distributed jobs: ``create_dataset`` should have
            run first, and every rank packing 960 h in parallel is not what
            anyone wants.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the view is missing and ``build_if_missing``
            is ``False``.

    Examples:
        >>> dataset = LibriSpeechDataset(split="test_clean")
        >>> sorted(dataset[0].keys())
        ['speech', 'text']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        archive_root: str | Path | None = None,
        source_dir: str | Path | None = None,
        build_if_missing: bool = False,
    ) -> None:
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self._recipe_root = recipe_root
        self._archive_root = archive_root
        self.view_path = split_view_path(recipe_root, self.split, archive_root)

        if not self.view_path.is_file():
            if not build_if_missing:
                raise FileNotFoundError(
                    f"Split view not found: {self.view_path}. "
                    "Run `python run.py --stages create_dataset ...` first."
                )
            builder = LibriSpeechBuilder()
            builder_kwargs = {
                "recipe_dir": recipe_root,
                "source_dir": source_dir,
                "archive_root": archive_root,
                "splits": [self.split],
            }
            if not builder.is_source_prepared(**builder_kwargs):
                builder.prepare_source(**builder_kwargs)
            builder.build(**builder_kwargs)

        view = read_view(self.view_path)
        self._utt_ids: List[str] = view["id"]
        self._sources: List[str] = view["source"]
        self._bin_index: List[int] = view["bin_index"]
        self._start_byte: List[int] = view["start_byte"]
        self._end_byte: List[int] = view["end_byte"]

        # One transcript table per raw archive the view touches.
        texts: dict[str, str] = {}
        for source in dict.fromkeys(self._sources):
            texts.update(
                read_text_table(raw_archive_dir(recipe_root, source, archive_root))
            )
        missing = [utt_id for utt_id in self._utt_ids if utt_id not in texts]
        if missing:
            raise RuntimeError(
                f"{len(missing)} utterance(s) in {self.view_path} have no "
                f"transcript (first: {missing[0]}). Rebuild the split."
            )
        self._texts: List[str] = [texts[utt_id] for utt_id in self._utt_ids]

    def __len__(self) -> int:
        return len(self._utt_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            from omniio.interface import audio_read
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_OMNIIO_HINT) from exc

        index = int(idx)
        archive_dir = raw_archive_dir(
            self._recipe_root, self._sources[index], self._archive_root
        )
        start = int(self._start_byte[index])
        read = audio_read(
            str(blob_path(archive_dir, self._bin_index[index])),
            start,
            int(self._end_byte[index]) - start,
        )
        array = np.asarray(read.array, dtype=np.float32)
        if array.ndim == 2 and array.shape[1] == 1:
            # LibriSpeech is mono; espnet2's frontend expects (samples,).
            array = array[:, 0]

        # No "utt_id" key on purpose: espnet2's CommonPreprocessor passes keys
        # it does not recognise through unchanged, and a str value then reaches
        # CommonCollateFn, which assumes every value is an array.
        return {"speech": array, "text": self._texts[index]}


def gather_training_text(
    recipe_dir: str | Path | None = None,
    archive_root: str | Path | None = None,
    split: str = "train_960",
    **_kwargs: Any,
) -> List[str]:
    """Collect transcripts for tokenizer training from a packed split.

    Reading the archive keeps the tokenizer text identical to what training
    consumes, and costs one sequential read of the view and the transcript
    tables it references.

    Args:
        recipe_dir: Recipe root used to resolve the archive path. Defaults to
            the current working directory.
        archive_root: Directory holding ``archive/``.
        split: Logical split whose transcripts are used.
        **_kwargs: Unused extra options for API compatibility.

    Returns:
        Transcript strings, in the split's own order.

    Raises:
        FileNotFoundError: If the split has not been built yet.
    """
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    view_path = split_view_path(recipe_root, str(split), archive_root)
    if not view_path.is_file():
        raise FileNotFoundError(
            f"Split view not found: {view_path}. "
            "Run `python run.py --stages create_dataset ...` first."
        )
    view = read_view(view_path)
    texts: dict[str, str] = {}
    for source in dict.fromkeys(view["source"]):
        texts.update(
            read_text_table(raw_archive_dir(recipe_root, source, archive_root))
        )
    return [texts[utt_id] for utt_id in view["id"]]
