"""Task-ready files of the URGENT 2025 SNOAR recipe.

Training mixes are made on the fly from the URGENT 2025 OMNIIO store, so the
builder writes only these files into ``out_dir``:

- the URGENT 2026 track-1 dev set resampled to 16 kHz (``dev16k/``) and its
  manifest ``manifest/dev.jsonl``, which ``infer`` and ``measure`` read;
- with ``commercial_only``, the per-file noise license record and the
  commercial-use lists (:mod:`egs3.urgent25.snoar.dataset.licenses`), which
  ``conf/training_supervised_commercial_bsrnn.yaml`` reads. The default pool
  (``conf/training_supervised_bsrnn.yaml``) uses every file and needs no
  license metadata.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from egs3.urgent25.snoar.dataset.licenses import (
    audit_noise_licenses,
    write_license_record,
)
from espnet3.components.data.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

LICENSE_FILES = (
    "noise_licenses.tsv",
    "noise_attribution.tsv",
    "commercial_allow.txt",
    "musan_noise_commercial.txt",
)
DEV_MANIFEST = "manifest/dev.jsonl"


def read_store_audio(row: dict, fs: int) -> np.ndarray:
    """Read channel 0 of one OMNIIO store row, resampled to ``fs``.

    Args:
        row: Row of ``metadata.parquet`` (``path``, ``start_byte``,
            ``end_byte``).
        fs: Output sampling rate.

    Returns:
        Float32 samples.
    """
    with open(row["path"], "rb") as f:
        f.seek(int(row["start_byte"]))
        data = f.read(int(row["end_byte"]) - int(row["start_byte"]))
    x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x[:, 0]
    if rate != fs:
        g = np.gcd(int(fs), int(rate))
        x = resample_poly(x, fs // g, rate // g).astype(np.float32)
    return x


def write_dev_set(dev_root: Path, out_dir: Path, fs: int = 16000) -> int:
    """Write the URGENT 2026 dev set at ``fs`` and its manifest.

    Args:
        dev_root: OMNIIO store of URGENT 2026 with ``dev_noisy`` and
            ``dev_clean`` subsets.
        out_dir: Output directory; FLAC files go to ``dev16k/{noisy,clean}``.
        fs: Output sampling rate.

    Returns:
        The number of recordings written.

    Raises:
        ValueError: If a noisy recording has no clean reference.
    """
    import pyarrow.parquet as pq

    clean = {
        r["id"]: r
        for r in pq.read_table(dev_root / "dev_clean" / "metadata.parquet").to_pylist()
    }
    noisy = pq.read_table(dev_root / "dev_noisy" / "metadata.parquet").to_pylist()
    missing = [r["id"] for r in noisy if r["id"] not in clean]
    if missing:
        raise ValueError(
            f"{len(missing)} dev recordings lack a reference: {missing[:3]}"
        )
    for kind in ("noisy", "clean"):
        (out_dir / "dev16k" / kind).mkdir(parents=True, exist_ok=True)
    manifest = out_dir / DEV_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        for row in noisy:
            paths = {}
            for kind, src in (("noisy", row), ("clean", clean[row["id"]])):
                paths[kind] = out_dir / "dev16k" / kind / f"{row['id']}.flac"
                sf.write(paths[kind], read_store_audio(src, fs), fs)
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "audio": str(paths["noisy"]),
                        "speech_ref": str(paths["clean"]),
                    }
                )
                + "\n"
            )
    return len(noisy)


class Urgent25Builder(DatasetBuilder):
    """Write the 16 kHz dev set and, optionally, the license lists.

    Every method takes the ``create_dataset`` section of the training config
    as keyword arguments: ``urgent_root``, ``dev_root``, ``out_dir`` and
    ``commercial_only``. With ``commercial_only`` the license audit also
    needs ``fsd50k_meta``, ``fma_meta`` and optionally ``fsdkaggle_meta`` and
    ``musan_noise``.
    """

    def is_source_prepared(
        self, urgent_root: Optional[str] = None, dev_root: Optional[str] = None, **_
    ) -> bool:
        """Return whether both OMNIIO stores are present."""
        return bool(urgent_root and dev_root) and all(
            (Path(r) / sub / "metadata.parquet").is_file()
            for r, sub in ((urgent_root, "noise_fsd50k"), (dev_root, "dev_noisy"))
        )

    def prepare_source(
        self, urgent_root: Optional[str] = None, dev_root: Optional[str] = None, **_
    ) -> None:
        """Explain where the stores come from.

        Raises:
            FileNotFoundError: Always (the stores cannot be made here).
        """
        raise FileNotFoundError(
            f"URGENT stores not found (URGENT25_ROOT={urgent_root}, "
            f"URGENT26_ROOT={dev_root}). Prepare the URGENT 2025 training data "
            "and the URGENT 2026 track-1 dev set with the challenge scripts "
            "(https://github.com/urgent-challenge) and pack them as OMNIIO "
            "stores (<subset>/metadata.parquet + blobs)."
        )

    def is_built(
        self, out_dir: Optional[str] = None, commercial_only: bool = False, **_
    ) -> bool:
        """Return whether the dev manifest (and license lists) exist."""
        if not out_dir:
            return False
        files = (DEV_MANIFEST,) + (LICENSE_FILES if commercial_only else ())
        return all((Path(out_dir) / f).is_file() for f in files)

    def build(
        self,
        urgent_root: Optional[str] = None,
        dev_root: Optional[str] = None,
        out_dir: Optional[str] = None,
        fsd50k_meta: Optional[str] = None,
        fma_meta: Optional[str] = None,
        fsdkaggle_meta: Optional[str] = None,
        musan_noise: Optional[str] = None,
        commercial_only: bool = False,
        **_,
    ) -> None:
        """Write the dev set and, with ``commercial_only``, run the audit.

        Raises:
            ValueError: If ``out_dir`` is unset, or ``commercial_only`` is set
                without ``fsd50k_meta`` and ``fma_meta``.
        """
        if not out_dir:
            raise ValueError("create_dataset.out_dir must be set")
        out = Path(out_dir)
        if commercial_only and not all((out / f).is_file() for f in LICENSE_FILES):
            if not (fsd50k_meta and fma_meta):
                raise ValueError(
                    "commercial_only needs create_dataset.fsd50k_meta and "
                    "fma_meta: extract "
                    "FSD50K.metadata.zip (https://zenodo.org/records/4060432) and "
                    "fma_metadata.zip (https://github.com/mdeff/fma)"
                )
            rows = audit_noise_licenses(
                Path(urgent_root),
                Path(fsd50k_meta),
                Path(fma_meta),
                Path(fsdkaggle_meta) if fsdkaggle_meta else None,
                Path(musan_noise) if musan_noise else None,
            )
            logger.info("license audit: %s", write_license_record(rows, out))
        if not (out / DEV_MANIFEST).is_file():
            n = write_dev_set(Path(dev_root), out)
            logger.info("dev set: %d recordings -> %s", n, out / DEV_MANIFEST)
