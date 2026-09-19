"""SLURP dataset served from the recipe-local TSV manifests."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.slurp.slu.dataset.builder import (
    ensure_built,
    get_manifest_path,
    read_manifest,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CONFIG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CONFIG["supported_splits"]}


class SlurpDataset(TorchDataset):
    """SLURP served as an intent-prefixed ASR task.

    Each sample's ``text`` is ``"<scenario>_<action> <transcript>"``, so a plain
    attention/CTC ASR model predicts the intent as its first token and the
    transcript after it. This is the formulation of ``egs2/slurp/asr1``, and it
    is why the recipe needs no SLU-specific model or task class.

    Which fields a sample carries depends on the task the config selects:

    - ``with_transcript=False`` (default, the ASR configs): ``speech`` and
      ``text`` only -- the two fields ``CommonPreprocessor`` and the ASR collate
      function accept.
    - ``with_transcript=True`` (the SLU config): additionally ``transcript``,
      the words on their own, which ``espnet2.tasks.slu.SLUTask`` tokenizes
      separately and its BERT post-decoder reads.

    No ``utt_id`` is added in either case: the sample dictionary is passed
    onward by the dataset pipeline, where an unsupported field breaks
    collation, so inference identifies items by their index and
    ``src/inference.py`` writes that index into the SCP files.

    Args:
        split: One of ``train``, ``train_synthetic``, ``devel`` or ``test``.
        recipe_dir: Recipe directory. Defaults to the directory holding this
            module, which is correct whenever the recipe is used in place.
        source_dir: Optional corpus root override, only consulted when the
            manifests still have to be built.
        with_transcript: Whether to also return the transcript. Set it from
            ``data_src_args`` in the SLU config; leave it off for ASR, whose
            collate function cannot take the extra field.
        transcript_source: Where that transcript comes from. ``None`` (default)
            uses the corpus transcript from the manifest, the ground-truth
            setting -- ``--gt true`` in ``egs2/slurp/slu1/local/data.sh``. A
            directory instead makes the dataset read
            ``<transcript_source>/<split>/hyp_transcript.scp``, the first-pass
            ASR hypotheses dumped by the ``infer`` stage, which is the
            ``--gt false`` setting: the SLU model then trains on the same kind
            of recognition errors it will meet at test time. The SCP is keyed by
            item index and read in manifest order, so it must come from a run
            over this same split.

    Raises:
        ValueError: If ``split`` is not a known SLURP split.
        FileNotFoundError: If the manifests are missing and the corpus cannot be
            found to build them.

    Examples:
        >>> dataset = SlurpDataset(split="devel")
        >>> sorted(dataset[0])
        ['speech', 'text']
        >>> sorted(SlurpDataset(split="devel", with_transcript=True)[0])
        ['speech', 'text', 'transcript']
        >>> dataset[0]["text"].split(maxsplit=1)[0]
        'calendar_query'
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        with_transcript: bool = False,
        transcript_source: str | Path | None = None,
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
        ensure_built(recipe_root, source_dir=source_dir)

        self.with_transcript = bool(with_transcript)
        self._rows = read_manifest(get_manifest_path(recipe_root, self.split))
        self._transcripts = (
            read_hypothesis_transcripts(
                Path(transcript_source) / self.split / "hyp_transcript.scp",
                expected=len(self._rows),
            )
            if transcript_source is not None
            else None
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[int(idx)]
        array, _sampling_rate = sf.read(row["wav_path"])
        sample = {
            "speech": np.asarray(array, dtype=np.float32),
            "text": f"{row['intent']} {row['transcript']}",
        }
        if self.with_transcript:
            sample["transcript"] = (
                row["transcript"]
                if self._transcripts is None
                else self._transcripts[int(idx)]
            )
        return sample


def read_hypothesis_transcripts(path: Path, expected: int) -> list[str]:
    """Read first-pass ASR transcripts, in the item order the manifest defines.

    Also used by ``src/tokenizer.py`` to build the transcript token list from
    the same text the ASR-transcript config trains on.

    Args:
        path: ``<transcript_source>/<split>/hyp_transcript.scp``.
        expected: Number of manifest rows the SCP has to match.

    Returns:
        One transcript per item, in manifest order.

    Raises:
        FileNotFoundError: If the SCP is missing, meaning the `infer` stage that
            dumps it has not run for this split.
        RuntimeError: If it holds a different number of rows than the manifest,
            which means it came from a run over a different split or a run that
            was cut short.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"First-pass transcripts not found: {path}. Run the infer stage "
            "with conf/inference_transcripts.yaml first."
        )
    transcripts: list[str] = []
    with path.open("r", encoding="utf-8") as scp_file:
        for line in scp_file:
            parts = line.rstrip("\n").split(maxsplit=1)
            if not parts:
                continue
            transcripts.append(parts[1] if len(parts) > 1 else "")
    if len(transcripts) != expected:
        raise RuntimeError(
            f"{path} holds {len(transcripts)} transcripts but the manifest has "
            f"{expected}."
        )
    return transcripts
