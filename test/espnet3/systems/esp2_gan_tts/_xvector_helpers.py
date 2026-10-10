"""Shared helpers for the x-vector provider/runner tests."""

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from espnet3.systems.esp2_gan_tts.xvector_provider import XVectorProvider


def write_manifest(tmp_path, rows, name="train.tsv"):
    """Write TSV rows to ``tmp_path/name`` and return the path."""
    path = tmp_path / name
    path.write_text("".join(rows), encoding="utf-8")
    return path


def write_wav(tmp_path, name="a.wav", seconds=1.0, sr=16000):
    """Write a silent wav of ``seconds`` at ``sr`` and return the path."""
    path = tmp_path / name
    sf.write(path, np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return path


def make_config(**xvector):
    """Build a training config whose ``xvector`` block is ``xvector``."""
    return OmegaConf.create({"xvector": xvector or {"toolkit": "speechbrain"}})


def make_provider(manifest, tmp_path, **xvector):
    """Build an XVectorProvider for ``manifest`` writing under ``tmp_path``."""
    return XVectorProvider(
        make_config(**xvector),
        params={
            "manifest_path": str(manifest),
            "output_dir": str(tmp_path / "xvec"),
        },
    )
