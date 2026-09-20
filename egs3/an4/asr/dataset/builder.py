"""Prepare the full AN4 corpus with the ESPnet2 split and speed perturbation."""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

RECIPE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = OmegaConf.to_container(
    load_config_with_defaults(
        str(Path(__file__).with_name("config.yaml")), resolve=False
    )["builder"],
    resolve=True,
)
SPLITS = ("train", "valid", "test")


def atomic_write(path: Path, text: str) -> None:
    """Replace a text artifact after its complete contents are written.

    Args:
        path: Destination file; missing parent directories are created.
        text: UTF-8 text to publish.

    Raises:
        OSError: Writing or atomically replacing the destination fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def read_manifest(path: Path) -> list[tuple[str, Path, str]]:
    """Read the ordered utterance ID, waveform path and transcript rows.

    Args:
        path: Builder-produced TSV, with no header and exactly three fields.

    Returns:
        List of (utterance ID, Path, transcript) tuples in file order.

    Raises:
        ValueError: Rows are malformed, empty or contain duplicate IDs.
        FileNotFoundError: The manifest has not been prepared.
    """
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            uid, waveform, text = line.rstrip("\n").split("\t", maxsplit=2)
            rows.append((uid, Path(waveform), text))
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate utterance IDs: {path}")
    return rows


class An4Builder:
    """Download AN4 and build manifests; checks do not create any files.

    Builder methods accept ``recipe_dir``, optional ``source_dir`` (the extracted
    AN4 directory), and ``dev_size``. The latter defaults to the original 100
    utterances and can be reduced for synthetic tests.
    """

    @staticmethod
    def _paths(recipe_dir=None, source_dir=None):
        root = Path(recipe_dir or RECIPE_ROOT).resolve()
        source = Path(source_dir).resolve() if source_dir else root / "downloads/an4"
        return root, source

    def is_source_prepared(self, recipe_dir=None, source_dir=None, **kwargs):
        """Check the source layout without changing the filesystem.

        Args:
            recipe_dir: Recipe root, or None for the installed recipe.
            source_dir: Extracted AN4 root, or None for downloads/an4.
            **kwargs: Unused stage settings.

        Returns:
            Whether both original transcript files and waveform directories exist.
        """
        _, source = self._paths(recipe_dir, source_dir)
        return all(
            (source / relative).exists()
            for relative in (
                "etc/an4_train.transcription",
                "etc/an4_test.transcription",
                "wav/an4_clstk",
                "wav/an4test_clstk",
            )
        )

    def prepare_source(self, recipe_dir=None, source_dir=None, **kwargs):
        """Download/extract AN4, or validate an explicit source directory.

        Args:
            recipe_dir: Recipe root; defaults to this recipe's installed directory.
            source_dir: Extracted AN4 root, or None to use downloads/an4.
            **kwargs: Unused builder settings accepted by stage dispatch.

        Raises:
            FileNotFoundError: An explicitly supplied source is incomplete.
            RuntimeError: The extracted archive lacks required AN4 files.
        """
        root, _ = self._paths(recipe_dir, source_dir)
        if self.is_source_prepared(recipe_dir, source_dir):
            return
        if source_dir is not None:
            raise FileNotFoundError(f"Incomplete AN4 source directory: {source_dir}")
        archive = root / "downloads/an4_sphere.tar.gz"
        if not archive.is_file():
            temporary = archive.with_suffix(".part")
            download_url(CONFIG["url"], temporary)
            temporary.replace(archive)
        extract_targz(archive, archive.parent)
        if not self.is_source_prepared(recipe_dir, source_dir):
            raise RuntimeError("AN4 archive does not contain the expected corpus files")

    def _signature(self, recipe_dir=None, source_dir=None, dev_size=None, **kwargs):
        _, source = self._paths(recipe_dir, source_dir)
        return {
            **CONFIG,
            "source_dir": str(source),
            "dev_size": CONFIG["dev_size"] if dev_size is None else dev_size,
            "format_version": 4,
        }

    def is_built(self, recipe_dir=None, source_dir=None, dev_size=None, **kwargs):
        """Check prepared manifests against the source and builder settings.

        Args:
            recipe_dir: Recipe root, or None for the installed recipe.
            source_dir: Optional explicit extracted AN4 root.
            dev_size: Validation size, or None for the source default of 100.
            **kwargs: Unused stage settings.

        Returns:
            Whether complete current-format preparation is recorded.
        """
        root, _ = self._paths(recipe_dir, source_dir)
        marker = root / "data/manifest/build.json"
        if not marker.is_file() or not all(
            (root / f"data/manifest/{split}.tsv").is_file() for split in SPLITS
        ):
            return False
        return json.loads(marker.read_text()) == self._signature(
            recipe_dir, source_dir, dev_size
        )

    @staticmethod
    def _transcripts(source, split):
        subdir = "an4_clstk" if split == "train" else "an4test_clstk"
        rows = []
        transcript = source / f"etc/an4_{split}.transcription"
        for line in transcript.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"(.*?)\s+\(([^()]+)\)", line.strip())
            if match is None:
                raise ValueError(f"Malformed AN4 transcript: {line}")
            text, recording = match.groups()
            text = text.removeprefix("<s> ").removesuffix(" </s>")
            prefix, speaker, suffix = recording.split("-")
            uid = f"{speaker}-{prefix}-{suffix}"
            waveform = source / "wav" / subdir / speaker / f"{recording}.sph"
            rows.append((uid, waveform, text))
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate utterance IDs in {transcript}")
        return sorted(rows)

    def build(self, recipe_dir=None, source_dir=None, dev_size=None, **kwargs):
        """Split, augment and write ASR manifests plus original LM text.

        Args:
            recipe_dir: Destination recipe root, or None for this recipe.
            source_dir: Optional existing extracted AN4 root.
            dev_size: Validation count, defaulting to the source's first 100 IDs.
            **kwargs: Unused stage settings.

        Raises:
            FileNotFoundError: Raw data has not been prepared.
            ValueError: A split is empty, its size is invalid, or audio is not 16 kHz.
            RuntimeError: The source recipe's SoX dependency is unavailable.

        Notes:
            Writes data/wav, data/manifest and pre-filtering data/lm text.
            Rebuilding replaces generated artifacts; old statistics and tokenizers
            must be regenerated after any data or builder change.
        """
        root, source = self._paths(recipe_dir, source_dir)
        signature = self._signature(recipe_dir, source_dir, dev_size)
        if not self.is_source_prepared(recipe_dir, source_dir):
            raise FileNotFoundError("Run prepare_source before building AN4")
        if shutil.which("sox") is None:
            raise RuntimeError("SoX is required for the original speed perturbation")
        training = self._transcripts(source, "train")
        ndev = signature["dev_size"]
        if not 0 < ndev < len(training):
            raise ValueError(
                "dev_size must leave nonempty training and validation sets"
            )
        splits = {
            "valid": training[:ndev],
            "train": training[ndev:],
            "test": self._transcripts(source, "test"),
        }
        marker = root / "data/manifest/build.json"
        marker.unlink(missing_ok=True)
        for split, rows in splits.items():
            output = root / "data/wav" / split
            output.mkdir(parents=True, exist_ok=True)
            prepared = []
            lm_text = []
            for uid, path, text in rows:
                speech, rate = sf.read(path, dtype="int16", always_2d=True)
                if rate != CONFIG["sample_rate"]:
                    raise ValueError(f"Expected 16 kHz AN4 audio: {path} ({rate} Hz)")
                # ESPnet2's sph2pipe command selects channel 1.
                original = output / f"{uid}.wav"
                sf.write(original, speech[:, 0], rate, subtype="PCM_16")
                factors = CONFIG["speed_factors"] if split == "train" else [1.0]
                for factor in factors:
                    item_id = f"sp{factor:.1f}-{uid}" if factor != 1.0 else uid
                    waveform = original
                    if factor != 1.0:
                        waveform = output / f"{item_id}.wav"
                        subprocess.run(
                            [
                                "sox",
                                str(original),
                                "-r",
                                str(rate),
                                str(waveform),
                                "speed",
                                str(factor),
                            ],
                            check=True,
                        )
                    if split != "train" or text.strip():
                        lm_text.append((item_id, text))
                    duration = sf.info(waveform).duration
                    if split == "test" or (
                        CONFIG["min_duration"] < duration < CONFIG["max_duration"]
                        and text.strip()
                    ):
                        prepared.append((item_id, waveform, text))
            atomic_write(
                root / f"data/lm/{split}.txt",
                "".join(f"{uid} {text}\n" for uid, text in sorted(lm_text)),
            )
            if not prepared:
                raise ValueError(f"Empty AN4 split after duration filtering: {split}")
            atomic_write(
                root / f"data/manifest/{split}.tsv",
                "".join(
                    f"{uid}\t{path}\t{text}\n" for uid, path, text in sorted(prepared)
                ),
            )
        atomic_write(marker, json.dumps(signature, indent=2) + "\n")
