"""VoxLingua107 dataset builder."""

from __future__ import annotations

import os
from collections import defaultdict
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

from omegaconf import OmegaConf

from egs3.voxlingua107.esp2_lid.src.download import DownloadProvider, DownloadRunner
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url


def _load_builder_config() -> dict:
    """Read the recipe dataset settings without resolving experiment paths."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_PREPARING = ".voxlingua107.preparing"
_BUILDING = ".voxlingua107.building"
_EXCLUDED_TRAIN_DIRS = {str(name) for name in _CFG["excluded_train_dirs"]}
_ISO3_CODES = {
    "ab": "abk",
    "af": "afr",
    "am": "amh",
    "ar": "ara",
    "as": "asm",
    "az": "aze",
    "ba": "bak",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "bo": "bod",
    "br": "bre",
    "bs": "bos",
    "ca": "cat",
    "ceb": "ceb",
    "cs": "ces",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "eo": "epo",
    "es": "spa",
    "et": "est",
    "eu": "eus",
    "fa": "fas",
    "fi": "fin",
    "fo": "fao",
    "fr": "fra",
    "gl": "glg",
    "gn": "grn",
    "gu": "guj",
    "gv": "glv",
    "ha": "hau",
    "haw": "haw",
    "hi": "hin",
    "hr": "hrv",
    "ht": "hat",
    "hu": "hun",
    "hy": "hye",
    "ia": "ina",
    "id": "ind",
    "is": "isl",
    "it": "ita",
    "iw": "heb",
    "ja": "jpn",
    "jw": "jav",
    "ka": "kat",
    "kk": "kaz",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "la": "lat",
    "lb": "ltz",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lv": "lav",
    "mg": "mlg",
    "mi": "mri",
    "mk": "mkd",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "msa",
    "mt": "mlt",
    "my": "mya",
    "ne": "nep",
    "nl": "nld",
    "nn": "nno",
    "no": "nor",
    "oc": "oci",
    "pa": "pan",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sa": "san",
    "sco": "sco",
    "sd": "snd",
    "si": "sin",
    "sk": "slk",
    "sl": "slv",
    "sn": "sna",
    "so": "som",
    "sq": "sqi",
    "sr": "srp",
    "su": "sun",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tk": "tuk",
    "tl": "tgl",
    "tr": "tur",
    "tt": "tat",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "war": "war",
    "yi": "yid",
    "yo": "yor",
    "zh": "cmn",
}


def resolve_source_root(source_dir: str | Path | None = None) -> Path:
    """Resolve the VoxLingua107 source directory."""
    return (
        Path(source_dir or os.environ[str(_CFG["source_env_var"])])
        .expanduser()
        .resolve()
    )


def resolve_metadata_root(
    recipe_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve recipe-local manifests independently of the source audio."""
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    recipe_root = (
        Path(recipe_dir) if recipe_dir is not None else Path(__file__).parents[1]
    )
    return recipe_root.resolve() / str(_CFG["data_path"])


def _iter_audio(source_root: Path, split: str) -> list[tuple[str, Path]]:
    """List original WAVs, excluding development duplicates from training."""
    if split == "dev":
        split_root = source_root / "dev"
        language_dirs = [
            p for p in split_root.iterdir() if p.is_dir() and p.name in _ISO3_CODES
        ]
    elif split == "train":
        split_root = source_root
        language_dirs = [
            p
            for p in split_root.iterdir()
            if p.is_dir()
            and p.name in _ISO3_CODES
            and p.name not in _EXCLUDED_TRAIN_DIRS
        ]
    else:
        raise ValueError(f"Unknown split: {split}")

    dev_names = (
        {wav.name for wav in (source_root / "dev").rglob("*.wav")}
        if split == "train"
        else set()
    )
    entries = [
        (_ISO3_CODES[language_dir.name], wav.resolve())
        for language_dir in sorted(language_dirs)
        for wav in sorted(language_dir.rglob("*.wav"))
        if wav.name not in dev_names
    ]
    return sorted(entries, key=lambda item: str(item[1]))


def _has_audio(source_root: Path, split: str) -> bool:
    """Check whether the requested split contains at least one WAV."""
    split_root = source_root / "dev" if split == "dev" else source_root
    excluded = _EXCLUDED_TRAIN_DIRS if split == "train" else set()
    return any(
        next(language_dir.rglob("*.wav"), None) is not None
        for language_dir in split_root.iterdir()
        if language_dir.is_dir()
        and language_dir.name in _ISO3_CODES
        and language_dir.name not in excluded
    )


def _has_complete_training_data(source_root: Path) -> bool:
    """Check that every training language has extracted audio."""
    return all(
        (source_root / language).is_dir()
        and next((source_root / language).rglob("*.wav"), None) is not None
        for language in _ISO3_CODES
    )


def _has_complete_training_metadata(metadata_root: Path) -> bool:
    """Check that the prepared training inventory contains all source languages."""
    category2utt = metadata_root / "train" / "category2utt"
    if not category2utt.is_file():
        return False
    categories = {
        line.split(maxsplit=1)[0]
        for line in category2utt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return categories == set(_ISO3_CODES.values())


def _write_split(source_root: Path, metadata_root: Path, split: str) -> None:
    """Write source-local manifests and language/category mappings."""
    entries = _iter_audio(source_root, split)
    if not entries:
        raise RuntimeError(f"No WAV files found for VoxLingua107 split '{split}'.")

    split_dir = metadata_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    language_to_indices: dict[str, list[str]] = defaultdict(list)

    with (
        (split_dir / "manifest.tsv").open("w", encoding="utf-8") as manifest,
        (split_dir / "utt2lang").open("w", encoding="utf-8") as utt2lang,
    ):
        for index, (language, wav_path) in enumerate(entries):
            utt_id = f"{language}_{wav_path.stem}"
            manifest.write(f"{utt_id}\t{wav_path}\t{language}\n")
            utt2lang.write(f"{index} {language}\n")
            language_to_indices[language].append(str(index))

    lines = [
        f"{language} {' '.join(indices)}\n"
        for language, indices in sorted(language_to_indices.items())
    ]
    (split_dir / "lang2utt").write_text("".join(lines), encoding="utf-8")
    (split_dir / "category2utt").write_text("".join(lines), encoding="utf-8")


class VoxLingua107Builder(DatasetBuilder):
    """Build manifests and category metadata from an extracted VoxLingua107 tree."""

    def is_source_prepared(
        self, source_dir: str | Path | None = None, **_kwargs
    ) -> bool:
        """Check that train and development WAV files are present."""
        source_root = resolve_source_root(source_dir)
        if (source_root / _PREPARING).exists():
            return False
        if not source_root.is_dir() or not (source_root / "dev").is_dir():
            return False
        return _has_complete_training_data(source_root) and _has_audio(
            source_root, "dev"
        )

    def prepare_source(
        self,
        source_dir: str | Path | None = None,
        zip_urls_url: str | None = None,
        dev_zip_url: str | None = None,
        parallel=None,
        **_kwargs,
    ) -> None:
        """Download and unzip the official corpus as in ESPnet2 local/data.sh.

        Keep completed archives and resume partial downloads with wget. A marker
        prevents interrupted extraction from being treated as a prepared source.
        Audio is extracted unchanged; no cropping or resampling is performed.

        Args:
            source_dir: Corpus directory, or the VOXLINGUA107 environment variable.
            zip_urls_url: Optional override of the official training ZIP list.
            dev_zip_url: Optional override of the development ZIP URL.
            parallel: ESPnet3 parallel configuration, e.g. an HPC backend with
                n_workers. Defaults to local sequential execution.
            **_kwargs: Unused arguments from the common builder interface.

        Example:
            >>> builder.prepare_source(source_dir="/corpora/voxlingua107")
        """
        source_root = resolve_source_root(source_dir)
        if self.is_source_prepared(source_dir=source_root):
            return
        source_root.mkdir(parents=True, exist_ok=True)
        marker = source_root / _PREPARING
        marker.touch()
        url_list = source_root / "zip_urls.txt"
        if not url_list.is_file():
            temporary = source_root / "zip_urls.txt.part"
            download_url(zip_urls_url or _CFG["zip_urls_url"], temporary)
            temporary.replace(url_list)
        urls = url_list.read_text(encoding="utf-8").split()
        if not urls:
            raise ValueError(f"Empty VoxLingua107 URL list: {url_list}")
        # The published list contains training ZIPs only.
        if not any(Path(urlparse(url).path).name == "dev.zip" for url in urls):
            urls.append(dev_zip_url or _CFG["dev_zip_url"])
        tasks = []
        for url in urls:
            name = Path(urlparse(url).path).name
            destination = source_root / "dev" if name == "dev.zip" else source_root
            tasks.append(
                {
                    "url": url,
                    "path": str(source_root / name),
                    "extract_to": str(destination),
                }
            )
        set_parallel(parallel or OmegaConf.create({"env": "local"}))
        DownloadRunner(
            DownloadProvider(tasks), output_dir=source_root / ".download", resume=False
        )(range(len(tasks)))
        if not _has_complete_training_data(source_root) or not _has_audio(
            source_root, "dev"
        ):
            raise RuntimeError(f"Incomplete VoxLingua107 extraction: {source_root}")
        marker.unlink()

    def is_built(
        self,
        recipe_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether all manifests and category mappings exist."""
        metadata_root = resolve_metadata_root(recipe_dir, data_dir)
        if (metadata_root / _BUILDING).exists():
            return False
        required = ("manifest.tsv", "utt2lang", "lang2utt", "category2utt")
        return _has_complete_training_metadata(metadata_root) and all(
            (metadata_root / split / name).is_file()
            and (metadata_root / split / name).stat().st_size > 0
            for split in ("train", "dev")
            for name in required
        )

    def build(
        self,
        source_dir: str | Path | None = None,
        recipe_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Create manifests and mappings from an already prepared source.

        Args:
            source_dir: Extracted corpus root, or VOXLINGUA107 when omitted.
            recipe_dir: Recipe root used to resolve the metadata destination.
            data_dir: Explicit metadata destination, overriding recipe_dir.
            **_kwargs: Unused arguments from the common builder interface.

        Example:
            >>> builder.build(source_dir="/corpora/voxlingua107", data_dir="data/voxlingua107")

        Each split contains a tab-separated manifest (utterance ID, WAV path,
        language). Mapping keys use zero-based Dataset indices, for example::

            # utt2lang: Dataset index -> language
            0 eng
            1 jpn
            # lang2utt and category2utt: language -> Dataset indices
            eng 0
            jpn 1

        lang2utt defines language order; category2utt groups category-sampler
        inputs. collect_stats regenerates these mappings with global indices
        when datasets or speed variants are combined.
        """
        source_root = resolve_source_root(source_dir)
        metadata_root = resolve_metadata_root(recipe_dir, data_dir)
        metadata_root.mkdir(parents=True, exist_ok=True)
        marker = metadata_root / _BUILDING
        marker.touch()
        for split in ("train", "dev"):
            _write_split(source_root, metadata_root, split)
        marker.unlink()
