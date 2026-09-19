"""MuST-C speech translation dataset implementation.

Reads segment-level speech translation examples directly from the raw
MuST-C v1 release layout::

    <lang-pair>/data/<split>/txt/<split>.yaml   # one flow-mapping entry per line:
                                                 # {duration, offset, speaker_id, wav}
    <lang-pair>/data/<split>/txt/<split>.en      # source (English) transcript, one
                                                  # line per yaml entry, same order
    <lang-pair>/data/<split>/txt/<split>.<tgt>   # target-language translation, same
    <lang-pair>/data/<split>/wav/<talk>.wav       # per-talk audio; segments are
                                                  # sliced out with an offset/duration

No Kaldi data prep is required. Unlike the egs2 recipe, no Moses punctuation
normalization / tokenization is applied here; only the per-side case
conventions of ``egs2/must_c/st1/run.sh`` (``src_case=lc.rm``, ``tgt_case=tc``)
are reproduced, because those change which characters exist at all.

Scope: the egs2 ST recipe (``egs2/must_c/st1``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from torch.utils.data import Dataset as TorchDataset

from egs3.must_c.st.dataset.builder import (
    LANG_PAIR,
    REQUIRED_SPLITS,
    SRC_LANG,
    TGT_LANG,
    VERSION,
    MustCSTBuilder,
    available_target_languages,
    kept_indices,
    resolve_source_root,
)
from espnet3.systems.st.text_case import apply_case as _apply_case
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

SPLIT_ALIASES: dict[str, str] = {
    str(k): str(v) for k, v in _DATASET_CFG["split_aliases"].items()
}
_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}

_COMPACT_SEGMENT_RE = re.compile(
    r"^-\s*\{\s*duration:\s*([^,}]+),\s*offset:\s*([^,}]+),"
    r".*?speaker_id:\s*([^,}]+),\s*wav:\s*([^,}]+)\s*\}\s*$"
)


@dataclass(frozen=True)
class MustCExample:
    """Internal index entry for one MuST-C segment."""

    utt_id: str
    wav_path: Path
    offset: float
    duration: float
    speaker_id: str
    src_text: str
    tgt_text: str
    src_lang: str
    tgt_lang: str


def _parse_segments(split_dir: Path, split: str) -> list[tuple[float, float, str, str]]:
    """Parse ``txt/<split>.yaml`` into (offset, duration, speaker_id, wav) tuples."""
    yaml_path = split_dir / "txt" / f"{split}.yaml"
    segments: list[tuple[float, float, str, str]] = []
    with yaml_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            match = _COMPACT_SEGMENT_RE.match(line)
            if match is not None:
                duration, offset, speaker_id, wav = match.groups()
                segments.append(
                    (float(offset), float(duration), speaker_id.strip(), wav.strip())
                )
                continue
            try:
                entry = yaml.safe_load(line)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Unrecognized MuST-C yaml entry in {yaml_path}: {line}"
                ) from exc
            # A line beginning with ``-`` is parsed as a one-item sequence by
            # PyYAML; unwrap that sequence to its mapping entry.
            if isinstance(entry, list) and len(entry) == 1:
                entry = entry[0]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Unrecognized MuST-C yaml entry in {yaml_path}: {line}"
                )
            required = {"duration", "offset", "speaker_id", "wav"}
            if not required.issubset(entry):
                raise ValueError(
                    f"Unrecognized MuST-C yaml entry in {yaml_path}: {line}"
                )
            try:
                duration = float(entry["duration"])
                offset = float(entry["offset"])
                speaker_id = str(entry["speaker_id"]).strip()
                wav = str(entry["wav"]).strip()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid MuST-C yaml entry in {yaml_path}: {line}"
                ) from exc
            segments.append((offset, duration, speaker_id, wav))
    return segments


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def _scan_split(
    lang_pair_root: Path, alias: str, tgt_lang: str = TGT_LANG
) -> list[MustCExample]:
    """Build an index for one split by zipping the yaml, en, and tgt files."""
    split_dir = lang_pair_root / "data" / alias
    wav_dir = split_dir / "wav"

    segments = _parse_segments(split_dir, alias)
    src_lines = _read_lines(split_dir / "txt" / f"{alias}.{SRC_LANG}")
    tgt_lines = _read_lines(split_dir / "txt" / f"{alias}.{tgt_lang}")

    if not (len(segments) == len(src_lines) == len(tgt_lines)):
        raise RuntimeError(
            f"MuST-C {alias}: yaml/{SRC_LANG}/{TGT_LANG} line counts differ "
            f"({len(segments)}, {len(src_lines)}, {len(tgt_lines)})"
        )

    examples: list[MustCExample] = []
    talk_counters: dict[str, int] = {}
    for (offset, duration, speaker_id, wav_name), src_text, tgt_text in zip(
        segments, src_lines, tgt_lines
    ):
        talk_id = Path(wav_name).stem
        idx = talk_counters.get(talk_id, 0)
        talk_counters[talk_id] = idx + 1
        utt_id = f"{alias}_{talk_id}_{idx:04d}"
        examples.append(
            MustCExample(
                utt_id=utt_id,
                wav_path=(wav_dir / wav_name).resolve(),
                offset=offset,
                duration=duration,
                speaker_id=speaker_id,
                src_text=src_text,
                tgt_text=tgt_text,
                src_lang=SRC_LANG,
                tgt_lang=tgt_lang,
            )
        )

    if not examples:
        raise RuntimeError(f"No segments found for MuST-C split: {split_dir}")
    return examples


@lru_cache(maxsize=None)
def _wav_samplerate(wav_path: str) -> int:
    return int(sf.info(wav_path).samplerate)


def _read_segment(wav_path: Path, offset: float, duration: float) -> np.ndarray:
    samplerate = _wav_samplerate(str(wav_path))
    start = int(round(offset * samplerate))
    frames = int(round(duration * samplerate))
    array, _sr = sf.read(
        str(wav_path), start=start, frames=frames, dtype="float32", always_2d=False
    )
    return np.asarray(array, dtype=np.float32)


class MustCSTDataset(TorchDataset):
    """Torch dataset that reads MuST-C from the original directory layout.

    Args:
        split: Logical split name, one of ``supported_splits`` in
            ``config.yaml`` (``train``, ``dev``, ``test``, ``tst-HE``). The
            logical ``test`` split is aliased to the physical ``tst-COMMON``
            split, matching the egs2 recipe's convention.
        recipe_dir: Optional recipe root. When omitted, defaults to this
            recipe's directory.
        source_dir: Optional override pointing at the raw corpus root (the
            directory that contains ``en-<tgt_lang>/``).
        task: ``"st"`` (default) returns the target-language translation as
            ``text``; ``"asr"`` returns the source-language transcript as
            ``text`` instead. ``src_text`` always carries the source side.
        src_case, tgt_case: Case conventions applied to each side, matching
            egs2's ``st.sh`` flags. Defaults reproduce
            ``egs2/must_c/st1/run.sh``: ``lc.rm`` for the source (lowercased,
            punctuation stripped) and ``tc`` for the target (truecased).

    Raises:
        ValueError: If ``split`` or ``task`` is unknown.
        FileNotFoundError: If the resolved source root or split directory
            does not exist.
        RuntimeError: If the yaml/text files are misaligned or empty.

    Examples:
        >>> dataset = MustCSTDataset(split="train")  # doctest: +SKIP
        >>> sample = dataset[0]  # doctest: +SKIP
        >>> sorted(sample.keys())  # doctest: +SKIP
        ['speech', 'src_text', 'text']
    """

    split_aliases = SPLIT_ALIASES

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        task: str = "st",
        tgt_lang: str | None = None,
        cache: dict | None = None,
        src_case: str = "lc.rm",
        tgt_case: str = "tc",
        apply_filter: bool = True,
        return_utt_id: bool = False,
    ) -> None:
        # False only for the cache builder, which indexes the corpus as released.
        self.apply_filter = bool(apply_filter)
        # OFF for training (a str breaks collation), ON for inference, which
        # reads samples one at a time and requires an id. See _sample.
        self.return_utt_id = bool(return_utt_id)
        self._keep: list[int] | None = None
        self.split = str(split)
        # egs2/must_c/st1/run.sh: src_case=lc.rm, tgt_case=tc
        self.src_case = str(src_case)
        self.tgt_case = str(tgt_case)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")
        if task not in {"asr", "st"}:
            raise ValueError("task must be 'asr' or 'st'")
        self.task = task
        self.tgt_lang = tgt_lang or TGT_LANG

        self._hf_cache = _load_hf_cache(cache, recipe_dir, self.split)
        if self._hf_cache is not None:
            required = {"audio_path", "src_text", "tgt_text", "offset", "duration"}
            if not required.issubset(self._hf_cache.column_names):
                raise RuntimeError(
                    "MuST-C HF cache predates segment metadata; recreate it with create_dataset"
                )
            # One columnar read, not 229,703 row reads.
            self._apply_duration_filter(self._hf_cache["duration"])
            return

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        builder = MustCSTBuilder()
        if not builder.is_source_prepared(
            recipe_dir=recipe_root, source_dir=source_dir, tgt_lang=self.tgt_lang
        ):
            builder.prepare_source(
                recipe_dir=recipe_root, source_dir=source_dir, tgt_lang=self.tgt_lang
            )

        alias = self.split_aliases.get(self.split, self.split)
        if self.tgt_lang == "all":
            self.lang_pair_root = None
            self._examples = [
                MustCExample(
                    f"{target}_{item.utt_id}",
                    item.wav_path,
                    item.offset,
                    item.duration,
                    item.speaker_id,
                    item.src_text,
                    item.tgt_text,
                    item.src_lang,
                    item.tgt_lang,
                )
                for target in available_target_languages(recipe_root, source_dir)
                for item in _scan_split(
                    resolve_source_root(recipe_root, source_dir, target), alias, target
                )
            ]
        else:
            self.lang_pair_root = resolve_source_root(
                recipe_root, source_dir=source_dir, tgt_lang=self.tgt_lang
            )
            split_dir = self.lang_pair_root / "data" / alias
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            self._examples = _scan_split(self.lang_pair_root, alias, self.tgt_lang)
        self._apply_duration_filter([e.duration for e in self._examples])

    def _apply_duration_filter(self, durations) -> None:
        """Drop segments st.sh stage 4 would have removed.

        Sets ``self._keep`` to the surviving positions, or leaves it ``None``
        when this split is not filtered (the test sets) or filtering is off.
        """
        if not self.apply_filter:
            return
        self._keep = kept_indices(durations, self.split)

    def _source_index(self, idx: int) -> int:
        """Map a dataset position onto the underlying corpus-order position."""
        if self._keep is None:
            return int(idx)
        return self._keep[int(idx)]

    def __len__(self) -> int:
        if self._keep is not None:
            return len(self._keep)
        if self._hf_cache is not None:
            return len(self._hf_cache)
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        index = self._source_index(idx)
        if self._hf_cache is not None:
            row = self._hf_cache[index]
            speech = _read_segment(
                Path(row["audio_path"]), float(row["offset"]), float(row["duration"])
            )
            src_text, tgt_text = str(row["src_text"]), str(row["tgt_text"])
            return self._sample(speech, src_text, tgt_text, str(row["utt_id"]))
        example = self._examples[index]
        speech = _read_segment(example.wav_path, example.offset, example.duration)
        return self._sample(speech, example.src_text, example.tgt_text, example.utt_id)

    def _sample(
        self, speech, src_text: str, tgt_text: str, utt_id: str | None = None
    ) -> dict[str, Any]:
        """Build one sample, applying each side's case convention.

        Only tensor-valued keys may be returned on the training path:
        ``CommonCollateFn`` pads and stacks every value, so a str raises
        ``AttributeError: 'str' object has no attribute 'dtype'`` on the first
        batch. (It is collation that rejects strings, not the preprocessor's
        ``Dict[str, np.ndarray]`` annotation -- typeguard samples collection
        elements and lets extra string keys through.) ``utt_id`` is therefore
        emitted only when ``return_utt_id`` is set, as inference does.
        """
        if self.task == "st":
            text, text_case = tgt_text, self.tgt_case
        else:
            text, text_case = src_text, self.src_case
        sample = {
            "speech": speech,
            "text": _apply_case(text, text_case),
            "src_text": _apply_case(src_text, self.src_case),
        }
        if self.return_utt_id and utt_id is not None:
            sample["utt_id"] = str(utt_id)
        return sample


def gather_training_text(
    recipe_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    side: str = "joint",
    case: str | None = None,
    **_kwargs,
) -> list[str]:
    """Collect train text for a SentencePiece model.

    Args:
        side: Which stream to return. ``"src"`` and ``"tgt"`` return one side,
            as ``STSystem`` needs for the two separate vocabularies egs2's
            ``st.sh`` builds (``src_nbpe``/``tgt_nbpe``). ``"joint"`` (default)
            concatenates both, for a single shared vocabulary.
        case: Case convention to apply, one of ``tc``, ``lc``, ``lc.rm``. When
            omitted it follows ``egs2/must_c/st1/run.sh``: ``lc.rm`` for the
            source side and ``tc`` for the target. A ``"joint"`` gather applies
            each side its own default.

    Returns:
        The requested text lines, in corpus order.
    """
    if side not in {"joint", "src", "tgt"}:
        raise ValueError(f"Unknown side {side!r}; expected joint, src or tgt")
    src_case = case or "lc.rm"
    tgt_case = case or "tc"

    cached = _load_hf_cache(_kwargs.get("cache"), recipe_dir, "train")
    if cached is not None:
        src = cached["src_text"] if side in {"joint", "src"} else []
        tgt = cached["tgt_text"] if side in {"joint", "tgt"} else []
    else:
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        lang_pair_root = resolve_source_root(recipe_root, source_dir=source_dir)
        examples = _scan_split(lang_pair_root, SPLIT_ALIASES.get("train", "train"))
        src = [e.src_text for e in examples] if side in {"joint", "src"} else []
        tgt = [e.tgt_text for e in examples] if side in {"joint", "tgt"} else []

    return [_apply_case(str(text), src_case) for text in src] + [
        _apply_case(str(text), tgt_case) for text in tgt
    ]


__all__ = [
    "MustCSTDataset",
    "gather_training_text",
    "LANG_PAIR",
    "SRC_LANG",
    "TGT_LANG",
    "VERSION",
    "REQUIRED_SPLITS",
]


def _load_hf_cache(cache, recipe_dir, split):
    import os

    environment_root = os.environ.get("EGS3_HF_CACHE_DIR")
    if cache is None and environment_root:
        cache = {"enabled": True, "backend": "hf", "cache_dir": environment_root}
    if not cache or not cache.get("enabled", False):
        return None
    from datasets import load_from_disk

    root = Path(cache.get("cache_dir", "data/hf"))
    if not root.is_absolute():
        root = Path(recipe_dir or Path.cwd()) / root
    split_root = root / "hf_audio_index" / str(split)
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"HF audio cache is missing: {split_root}. "
            "Run DatasetBuilder.build() first."
        )
    return load_from_disk(str(split_root))
