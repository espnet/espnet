"""LibriCSS segment dataset backed by stage-produced manifests.

The dataset reads per-recording JSON manifests written by the recipe's
``segment`` (``${exp_dir}/segments/<split>/<reco>.json``) or ``diarize``
(``${exp_dir}/diarized/<split>/<reco>.json``) stages:

.. code-block:: json

    {
      "reco": "session0_CH0_0L",
      "wav": "/abs/path/data/wavs/session0_CH0_0L.wav",
      "sample_rate": 16000,
      "segments": [
        {"utt_id": "...", "start": 8.53, "end": 24.63,
         "spk": "1", "text": "OPTIONAL ORACLE TRANSCRIPT"}
      ]
    }

Each item returns the audio slice plus flat metadata fields (``utt_id``,
``spk``, ``reco``, ``start``, ``end``, and ``text`` when the manifest carries
an oracle transcript). The ``infer`` stage only decodes ``speech``; the
metrics read the metadata back as ``dataset:<column>`` inputs (see
``conf/metrics*.yaml``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from espnet3.components.data.base_dataset import BaseDataset


class LibriCSSDataset(BaseDataset):
    """Dataset over one LibriCSS split's segment manifest directory."""

    fields = {
        "speech": np.ndarray,
        "utt_id": str,
        "spk": str,
        "reco": str,
        "start": float,
        "end": float,
        # Oracle transcript: only present in the oracle-segment flow.
        "text": (str, type(None)),
    }

    def __init__(
        self,
        split: str,
        manifest_dir: str | Path | None = None,
        recipe_dir: str | Path | None = None,
    ) -> None:
        """Index all manifest segments of one split.

        Args:
            split: Split name, e.g. ``dev`` or ``eval``.
            manifest_dir: Directory containing ``<split>/<reco>.json``
                manifests. Relative paths resolve against the recipe root.
            recipe_dir: Recipe root; defaults to this file's grandparent
                directory (``egs3/libricss/esp2_asr``).

        Raises:
            ValueError: If ``manifest_dir`` is missing.
            FileNotFoundError: If ``<manifest_dir>/<split>`` does not exist.
        """
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir
            else Path(__file__).resolve().parents[1]
        )
        if manifest_dir is None or str(manifest_dir) == "":
            raise ValueError(
                "LibriCSSDataset requires `manifest_dir` in data_src_args: "
                "the directory written by the `segment` or `diarize` stage, "
                "e.g. ${exp_dir}/diarized. Run those stages first."
            )
        manifest_root = Path(manifest_dir)
        if not manifest_root.is_absolute():
            manifest_root = recipe_root / manifest_root
        split_dir = manifest_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"No manifest directory for split '{split}': {split_dir}. "
                "Run the `segment` (and `diarize`) stages first, or point "
                "`manifest_dir` at the right experiment directory "
                "(it must match the exp_dir of the configs used)."
            )

        self.split = split
        self.manifest_dir = split_dir
        self.items: list[dict[str, Any]] = []
        self._sr_cache: dict[str, int] = {}
        self._frames_cache: dict[str, int] = {}

        for manifest_path in sorted(split_dir.glob("*.json")):
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            wav = manifest["wav"]
            reco = manifest["reco"]
            if "sample_rate" in manifest:
                self._sr_cache[wav] = int(manifest["sample_rate"])
            for seg in manifest["segments"]:
                item = dict(seg)
                item["wav"] = wav
                item["reco"] = reco
                if item.get("spk") is None:
                    item["spk"] = ""
                self.items.append(item)

        if not self.items:
            raise ValueError(f"Manifest directory {split_dir} contains no segments.")

    def _sample_rate(self, wav: str) -> int:
        """Return the sample rate of a wav, cached per path."""
        if wav not in self._sr_cache:
            self._sr_cache[wav] = int(sf.info(wav).samplerate)
        return self._sr_cache[wav]

    def _num_frames(self, wav: str) -> int:
        """Return the length of a wav in samples, cached per path."""
        if wav not in self._frames_cache:
            self._frames_cache[wav] = int(sf.info(wav).frames)
        return self._frames_cache[wav]

    def __len__(self) -> int:
        """Number of segments in this split."""
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one segment: audio slice plus flat metadata fields."""
        item = self.items[idx]
        wav = item["wav"]
        sr = self._sample_rate(wav)
        info_frames = self._num_frames(wav)

        start_sample = int(round(float(item["start"]) * sr))
        end_sample = int(round(float(item["end"]) * sr))
        start_sample = max(0, min(start_sample, info_frames - 1))
        frames = max(end_sample - start_sample, 1)
        frames = min(frames, info_frames - start_sample)

        audio, _ = sf.read(wav, start=start_sample, frames=frames, dtype="float32")
        speech = np.asarray(audio, dtype=np.float32).reshape(-1)

        out: dict[str, Any] = {
            "speech": speech,
            "utt_id": item.get("utt_id", f"{item['reco']}_{idx:08d}"),
            "spk": str(item.get("spk", "") or ""),
            "reco": item["reco"],
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        text = item.get("text")
        if text is not None:
            out["text"] = str(text)
        return out
