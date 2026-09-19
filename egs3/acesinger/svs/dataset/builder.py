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


def _song_id(segment_id: str) -> str:
    """Return the 4-digit Opencpop song id of ``acesinger_<singer>#<utt>``."""
    return segment_id.split("#", 1)[1][:4]


def _format_label(row: dict) -> str:
    """Render phoneme start/end times as the egs2 ``label`` line body."""
    return " ".join(
        f"{st:.3f} {et:.3f} {phn}"
        for st, et, phn in zip(row["phn_start_time"], row["phn_end_time"], row["phn"])
    )


def _format_score(row: dict) -> str:
    """Render the note sequence as a one-line JSON score."""
    notes = [
        [st, et, lyric, int(midi), phns]
        for st, et, lyric, midi, phns in zip(
            row["note_start_times"],
            row["note_end_times"],
            row["note_lyrics"],
            row["note_midi"],
            row["note_phns"],
        )
    ]
    score = dict(tempo=int(row["tempo"]), item_list=SCORE_ITEM_LIST, note=notes)
    return json.dumps(score, ensure_ascii=False)


def _convert_shard(
    parquet_path: str, wav_dir: str, fs: int, skip_songs: tuple[str, ...]
) -> list[tuple[str, str, str, int, str, str]]:
    """Write one parquet shard to wav files and return its manifest rows."""
    wav_root = Path(wav_dir)
    rows = []
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            utt_id = row["segment_id"]
            if _song_id(utt_id) in skip_songs:
                continue
            wav_path = wav_root / f"{utt_id}.wav"
            if not wav_path.exists():
                audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != fs:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=fs, res_type="soxr_hq"
                    )
                sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), fs, "PCM_16")
            rows.append(
                (
                    utt_id,
                    str(wav_path),
                    row["transcription"],
                    int(row["singer"]),
                    _format_label(row),
                    _format_score(row),
                )
            )
    return rows


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
        """Check whether every split has at least one parquet shard on disk."""
        return all(self._shards(recipe_dir, hf) for hf in _CFG["splits"].values())

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
        """Check whether all split manifests exist under ``data_path``."""
        data = Path(recipe_dir).resolve() / _CFG["data_path"]
        return all((data / p).is_file() for p in _CFG["manifest_paths"].values())

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Write wav files and TSV manifests for train, valid and test.

        Shards are converted in parallel with ``num_workers`` processes. Rows
        are sorted by utterance id so the manifest order is deterministic.
        """
        recipe_root = Path(recipe_dir).resolve()
        data = recipe_root / _CFG["data_path"]
        fs = int(_CFG["fs"])
        skip_songs = tuple(str(s) for s in _CFG.get("skip_songs", []))
        num_workers = int(_CFG["num_workers"])

        for split, hf_split in _CFG["splits"].items():
            shards = self._shards(recipe_root, hf_split)
            if not shards:
                raise RuntimeError(f"No parquet shard found for split '{hf_split}'")
            wav_dir = data / "wav" / split
            wav_dir.mkdir(parents=True, exist_ok=True)
            # Only the training split drops the skipped songs, as in egs2.
            split_skip = skip_songs if split == "train" else ()

            logger.info("Converting %d shard(s) of split '%s'", len(shards), split)
            rows = []
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                for shard_rows in pool.map(
                    _convert_shard,
                    [str(p) for p in shards],
                    [str(wav_dir)] * len(shards),
                    [fs] * len(shards),
                    [split_skip] * len(shards),
                ):
                    rows.extend(shard_rows)
            rows.sort(key=lambda r: r[0])

            manifest = data / _CFG["manifest_paths"][split]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write("\t".join(str(col) for col in row) + "\n")
            logger.info("Wrote %d utterances to %s", len(rows), manifest)
