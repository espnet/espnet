"""ACE-Opencpop dataset builder."""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# Score note layout written to the manifest, identical to the ``score.json``
# files that egs2/acesinger/svs1/local/data_prep.py produces with
# ``SingingScoreWriter``.
SCORE_ITEM_LIST = ["st", "et", "lyric", "midi", "phns"]


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def manifest_path(recipe_dir: str | Path, split: str) -> Path:
    """Return the manifest that ``build`` writes for ``split``.

    Raises:
        ValueError: If ``split`` is not one of the configured splits.
    """
    if split not in _CFG["manifest_paths"]:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of "
            f"{sorted(_CFG['manifest_paths'])}"
        )
    return Path(recipe_dir) / _CFG["data_path"] / _CFG["manifest_paths"][split]


def _song_id(segment_id: str) -> str:
    """Return the 4-digit Opencpop song id of ``acesinger_<singer>#<utt>``."""
    return segment_id.split("#", 1)[1][:4]


def _format_label(row: dict) -> str:
    """Render phoneme start/end times as the egs2 ``label`` line body."""
    fields = []
    for st, et, phn in zip(row["phn_start_time"], row["phn_end_time"], row["phn"]):
        fields.append(f"{st:.3f} {et:.3f} {phn}")
    return " ".join(fields)


def _format_score(row: dict) -> str:
    """Render the note sequence as a one-line JSON score."""
    notes = []
    for st, et, lyric, midi, phns in zip(
        row["note_start_times"],
        row["note_end_times"],
        row["note_lyrics"],
        row["note_midi"],
        row["note_phns"],
    ):
        notes.append([st, et, lyric, int(midi), phns])
    score = {"tempo": int(row["tempo"]), "item_list": SCORE_ITEM_LIST, "note": notes}
    return json.dumps(score, ensure_ascii=False)


def _write_wav(audio_bytes: bytes, wav_path: Path, fs: int) -> None:
    """Decode one Hub audio file and write it as 16-bit mono at ``fs`` Hz."""
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != fs:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=fs, res_type="soxr_hq")
    sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), fs, "PCM_16")


def _convert_shard(
    parquet_path: str, wav_dir: str, fs: int, skip_songs: list[str]
) -> list[str]:
    """Write the wav files of one parquet shard and return its manifest lines."""
    lines = []
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            utt_id = row["segment_id"]
            if _song_id(utt_id) in skip_songs:
                continue
            wav_path = Path(wav_dir) / f"{utt_id}.wav"
            if not wav_path.exists():
                _write_wav(row["audio"]["bytes"], wav_path, fs)
            fields = [
                utt_id,
                str(wav_path),
                row["transcription"],
                str(int(row["singer"])),
                _format_label(row),
                _format_score(row),
            ]
            lines.append("\t".join(fields) + "\n")
    return lines


class ACEOpencpopBuilder(DatasetBuilder):
    """Prepare and build ACE-Opencpop assets for the ESPnet3 SVS recipe.

    The source is the ``espnet/ace-opencpop-segments`` dataset on the Hugging
    Face Hub, which already holds the segmented singing, phoneme labels and
    music scores that ``egs2/acesinger/svs1/local/data_prep.py`` derives from
    the raw ACE-Studio renderings and Opencpop annotations. ``build`` turns its
    parquet shards into wav files plus one TSV manifest per split whose
    columns are ``utt_id``, ``wav_path``, ``text``, ``singer``, ``label`` and
    ``score`` (a one-line JSON in the egs2 ``score.json`` layout).
    """

    def _dataset_root(self, recipe_dir: str | Path) -> Path:
        return Path(recipe_dir).resolve() / _CFG["dataset_path"]

    def _shards(self, recipe_dir: str | Path, hf_split: str) -> list[Path]:
        return sorted((self._dataset_root(recipe_dir) / "data").glob(f"{hf_split}-*"))

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether every parquet shard of every split is on disk."""
        for hf_split in _CFG["splits"].values():
            shards = self._shards(recipe_dir, hf_split)
            if len(shards) == 0:
                return False
            # Shards are named <split>-<index>-of-<total>.parquet.
            total = int(shards[0].stem.split("-of-")[1])
            if len(shards) != total:
                return False
        return True

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Download the Hub dataset into ``dataset_path``."""
        from huggingface_hub import snapshot_download

        dataset_root = self._dataset_root(recipe_dir)
        dataset_root.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s", _CFG["hf_repo_id"], dataset_root)
        snapshot_download(
            _CFG["hf_repo_id"], repo_type="dataset", local_dir=str(dataset_root)
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether the manifests of all splits exist."""
        for split in _CFG["manifest_paths"]:
            if not manifest_path(Path(recipe_dir).resolve(), split).is_file():
                return False
        return True

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Write wav files and TSV manifests for train, valid and test.

        Shards are converted in parallel with ``num_workers`` processes. Rows
        are sorted by utterance id so the manifest order is deterministic.
        """
        recipe_dir = Path(recipe_dir).resolve()
        fs = int(_CFG["fs"])
        for split, hf_split in _CFG["splits"].items():
            shards = self._shards(recipe_dir, hf_split)
            if len(shards) == 0:
                raise RuntimeError(f"No parquet shard found for split '{hf_split}'")
            wav_dir = recipe_dir / _CFG["data_path"] / "wav" / split
            wav_dir.mkdir(parents=True, exist_ok=True)
            # Only the training split drops the skipped songs, as in egs2.
            skip_songs = []
            if split == "train":
                skip_songs = [str(song) for song in _CFG.get("skip_songs", [])]

            logger.info("Converting %d shard(s) of split '%s'", len(shards), split)
            lines = []
            with ProcessPoolExecutor(max_workers=int(_CFG["num_workers"])) as pool:
                futures = []
                for shard in shards:
                    future = pool.submit(
                        _convert_shard, str(shard), str(wav_dir), fs, skip_songs
                    )
                    futures.append(future)
                for future in futures:
                    lines.extend(future.result())
            # Each line starts with its utterance id.
            lines.sort()

            path = manifest_path(recipe_dir, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            logger.info("Wrote %d utterances to %s", len(lines), path)
