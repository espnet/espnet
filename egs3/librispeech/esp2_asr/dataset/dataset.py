"""LibriSpeech dataset backed by Omni-IO archives."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Any, List

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech.esp2_asr.dataset.builder import (
    METADATA_NAME,
    LibriSpeechBuilder,
    blob_path,
    read_text_table,
    split_archive_dir,
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


def _read_metadata(archive_dir: Path) -> dict[str, list]:
    """Read one archive's Parquet offset table as plain Python lists.

    Rows are sorted by utterance id so that the dataset order -- and therefore
    the statistics keys written by `collect_stats` -- does not depend on the
    order in which the blobs happened to be packed.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(archive_dir / METADATA_NAME)
    columns = table.to_pydict()
    order = sorted(range(len(columns["id"])), key=lambda i: columns["id"][i])
    return {
        "id": [columns["id"][i] for i in order],
        "bin_index": [columns["bin_index"][i] for i in order],
        "start_byte": [columns["start_byte"][i] for i in order],
        "end_byte": [columns["end_byte"][i] for i in order],
    }


class LibriSpeechDataset(TorchDataset):
    """Torch dataset that serves LibriSpeech out of Omni-IO archives.

    Args:
        split: Logical split name such as ``train_960`` or ``test_clean``.
        recipe_dir: Optional recipe root. Defaults to this recipe directory.
        archive_root: Directory holding ``archive/<split>/``. Normally set to
            ``${data_dir}`` from the config.
        source_dir: Optional LibriSpeech parent/root override, used only when
            the archive still has to be built.
        build_if_missing: Pack the archive when it does not exist yet. Keep
            this ``False`` for distributed jobs: ``create_dataset`` should have
            run first, and every rank packing 960 h in parallel is not what
            anyone wants.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the archive is missing and ``build_if_missing``
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
        self.archive_dir = split_archive_dir(recipe_root, self.split, archive_root)

        if not (self.archive_dir / METADATA_NAME).is_file():
            if not build_if_missing:
                raise FileNotFoundError(
                    f"Omni-IO archive not found: {self.archive_dir}. "
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

        metadata = _read_metadata(self.archive_dir)
        texts = read_text_table(self.archive_dir)
        missing = [utt_id for utt_id in metadata["id"] if utt_id not in texts]
        if missing:
            raise RuntimeError(
                f"{len(missing)} archived utterance(s) have no transcript in "
                f"{self.archive_dir} (first: {missing[0]}). Rebuild the split."
            )

        self._utt_ids: List[str] = metadata["id"]
        self._bin_index: List[int] = metadata["bin_index"]
        self._start_byte: List[int] = metadata["start_byte"]
        self._end_byte: List[int] = metadata["end_byte"]
        self._texts: List[str] = [texts[utt_id] for utt_id in self._utt_ids]

    def __len__(self) -> int:
        return len(self._utt_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            from omniio.interface import audio_read
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_OMNIIO_HINT) from exc

        index = int(idx)
        start = int(self._start_byte[index])
        read = audio_read(
            str(blob_path(self.archive_dir, self._bin_index[index])),
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

    Reading the archive's transcript table keeps the tokenizer text identical
    to what training consumes, and costs one sequential read.

    Args:
        recipe_dir: Recipe root used to resolve the archive path. Defaults to
            the current working directory.
        archive_root: Directory holding ``archive/<split>/``.
        split: Logical split whose transcripts are used.
        **_kwargs: Unused extra options for API compatibility.

    Returns:
        Transcript strings, ordered by utterance id.

    Raises:
        FileNotFoundError: If the split has not been packed yet.
    """
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    archive_dir = split_archive_dir(recipe_root, str(split), archive_root)
    texts = read_text_table(archive_dir)
    return [texts[utt_id] for utt_id in sorted(texts)]
