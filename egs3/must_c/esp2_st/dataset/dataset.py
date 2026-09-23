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

from importlib import resources
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset as TorchDataset

from egs3.must_c.esp2_st.dataset.builder import (
    LANG_PAIR,
    REQUIRED_SPLITS,
    SRC_LANG,
    TGT_LANG,
    VERSION,
    MustCSTBuilder,
    read_segment,
)
from espnet3.systems.esp2_st.normalization import apply_case as _apply_case
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

SPLIT_ALIASES: dict[str, str] = {
    str(k): str(v) for k, v in _DATASET_CFG["split_aliases"].items()
}
_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}

# Long/short filtering (egs2 st.sh stage 4). The HF cache is the unfiltered
# `${data_feats}/org/<dset>` side; the Dataset applies the bounds on read.
_FILTER_CFG = _CONFIG["filter"]
MIN_WAV_DURATION = float(_FILTER_CFG["min_wav_duration"])
MAX_WAV_DURATION = float(_FILTER_CFG["max_wav_duration"])
FILTERED_SPLITS: tuple[str, ...] = tuple(str(s) for s in _FILTER_CFG["splits"])


def split_is_filtered(split: str) -> bool:
    """Whether ``split`` is one st.sh would trim (train and valid, not test)."""
    return str(split) in FILTERED_SPLITS


def keep_duration(duration: float) -> bool:
    """Reproduce st.sh's ``$2 > min_length && $2 < max_length``.

    Both bounds are strict, as in the awk expression st.sh applies to
    ``utt2num_samples``.
    """
    return MIN_WAV_DURATION < float(duration) < MAX_WAV_DURATION


def kept_indices(durations, split: str) -> list[int] | None:
    """Indices of ``durations`` to keep for ``split``.

    Args:
        durations: Segment durations in seconds, in corpus order.
        split: Logical split name, e.g. ``"train"`` or ``"test"``.

    Returns:
        The positions to keep, or ``None`` when ``split`` is not filtered at
        all. ``None`` rather than ``list(range(len(durations)))`` so callers
        can skip the indirection entirely on the test splits.
    """
    if not split_is_filtered(split):
        return None
    return [i for i, duration in enumerate(durations) if keep_duration(duration)]


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
        """Index one MuST-C split, from the HF cache or the raw release."""
        if str(split) not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")
        if task not in {"asr", "st"}:
            raise ValueError("task must be 'asr' or 'st'")

        self.split = str(split)
        self.task = task
        self.tgt_lang = tgt_lang or TGT_LANG
        self.src_case = str(src_case)
        self.tgt_case = str(tgt_case)
        self.apply_filter = bool(apply_filter)
        self.return_utt_id = bool(return_utt_id)
        self._keep: list[int] | None = None

        self._hf_cache = _require_hf_cache(
            cache, recipe_dir, self.split, source_dir, self.tgt_lang
        )
        self._init_from_cache()

    def _init_from_cache(self) -> None:
        """Take durations from the cache column and filter on them."""
        required = {"audio_path", "src_text", "tgt_text", "offset", "duration"}
        if not required.issubset(self._hf_cache.column_names):
            raise RuntimeError(
                "MuST-C HF cache predates segment metadata; "
                "recreate it with create_dataset"
            )
        # One columnar read, not 229,703 row reads.
        self._apply_duration_filter(self._hf_cache["duration"])

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
        """Number of utterances, after the long/short filter if applied."""
        if self._keep is not None:
            return len(self._keep)
        return len(self._hf_cache)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample: speech, text, src_text, optionally utt_id."""
        row = self._hf_cache[self._source_index(idx)]
        speech = read_segment(
            Path(row["audio_path"]), float(row["offset"]), float(row["duration"])
        )
        return self._sample(
            speech, str(row["src_text"]), str(row["tgt_text"]), str(row["utt_id"])
        )

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
    tgt_lang: str | None = None,
    **_kwargs,
) -> list[str]:
    """Collect train text for a SentencePiece model.

    Args:
        side: Which stream to return. ``"src"`` and ``"tgt"`` return one side,
            as ``STSystem`` needs for the two separate vocabularies egs2's
            ``st.sh`` builds (``src_nbpe``/``tgt_nbpe``). ``"joint"`` (default)
            concatenates both, for a single shared vocabulary.
        tgt_lang: Target language of the pair to read. Only used when no HF
            cache is configured; the module default is ``all``, which is not a
            directory on disk.
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

    cached = _require_hf_cache(
        _kwargs.get("cache"), recipe_dir, "train", source_dir, tgt_lang or TGT_LANG
    )
    # The cache is unfiltered, which is what egs2 wants: run.sh:48-49 points
    # bpe_train_text at data/${train_set}, before st.sh stage 4 trims it.
    src = cached["src_text"] if side in {"joint", "src"} else []
    tgt = cached["tgt_text"] if side in {"joint", "tgt"} else []

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


def _require_hf_cache(cache, recipe_dir, split, source_dir=None, tgt_lang=None):
    """Return one split of the HF cache, building it first when it is absent.

    The cache is not optional: reading MuST-C straight from the release means
    re-parsing 229,703 yaml entries per Dataset construction (185s against 21s
    for train), and it skips the build-time decode check that turns a corrupt
    segment into a `failures.jsonl` entry instead of a crash mid-epoch.

    Raises:
        RuntimeError: If no cache is configured.
    """
    cached = _load_hf_cache(cache, recipe_dir, split, missing_ok=True)
    if cached is not None:
        return cached
    if not cache or not cache.get("enabled", False):
        raise RuntimeError(
            "must_c/esp2_st reads its splits from an HF cache. Enable it in the "
            "training config (`cache.enabled: true` with a `cache.cache_dir`), "
            "or run `--stages create_dataset` to build it."
        )
    builder = MustCSTBuilder()
    builder.build(
        recipe_dir=recipe_dir, cache=cache, source_dir=source_dir, tgt_lang=tgt_lang
    )
    return _load_hf_cache(cache, recipe_dir, split)


def _load_hf_cache(cache, recipe_dir, split, missing_ok: bool = False):
    """Load one split of the HF cache, or None when it is unavailable."""
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
        if missing_ok:
            return None
        raise FileNotFoundError(
            f"HF audio cache is missing: {split_root}. "
            "Run DatasetBuilder.build() first."
        )
    return load_from_disk(str(split_root))
