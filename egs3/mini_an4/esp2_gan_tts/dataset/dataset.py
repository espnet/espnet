"""Mini AN4 dataset returning text/speech/spembs items for the GAN-TTS system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from egs3.mini_an4.esp2_asr.dataset.dataset import ManifestEntry, MiniAn4Dataset
from egs3.mini_an4.esp2_gan_tts.dataset.builder import resolve_manifest_path
from espnet3.contract import Field


class MiniAn4GANTTSDataset(MiniAn4Dataset):
    """Mini AN4 items for GAN-TTS training and inference.

    The manifests are the ASR recipe's with a speaker column added, built by
    ``MiniAn4GANTTSBuilder`` under this recipe's directory. Each item carries
    the transcript under ``text``, the 16 kHz waveform under ``speech`` when
    ``load_speech``, and the speaker
    embedding the ``compute_xvectors`` stage wrote for the utterance under
    ``spembs`` when ``load_xvector``. In ``inference`` mode it also carries
    ``utt_id``, which heads every inference output, and ``wav_path`` and
    ``raw_text``, which the ``measure`` stage reads as ``dataset:wav_path`` and
    ``dataset:raw_text``.

    Example:
        .. code-block:: python

            >>> xvector_dir = "data/x_vectors/voxcelebs12_rawnet3_train"
            >>> dataset = MiniAn4GANTTSDataset(split="train", xvector_dir=xvector_dir)
            >>> sorted(dataset[0])
            ['speech', 'spembs', 'text']
            >>> plain = MiniAn4GANTTSDataset(split="train", load_xvector=False)
            >>> sorted(plain[0])
            ['speech', 'text']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_speech: bool = True,
        load_xvector: bool = True,
        xvector_dir: str | Path | None = None,
        inference: bool = False,
    ) -> None:
        """Build the AN4 manifests under this recipe and load one split.

        Args:
            split: ``train``, ``valid`` or ``test``.
            recipe_dir: Recipe root; defaults to this recipe's directory.
            manifest_path: Manifest to read instead of the split's own, such
                as the one the ``remove_long_short`` stage wrote. Relative
                paths resolve against ``recipe_dir``.
            load_speech: Load the waveform into ``speech``.
            load_xvector: Load the speaker embedding into ``spembs``.
            xvector_dir: Directory of ``<utt_id>.pt`` embeddings written by
                the ``compute_xvectors`` stage. Required when
                ``load_xvector``. Relative paths resolve against
                ``recipe_dir``.
            inference: Add ``utt_id``, ``wav_path`` and ``raw_text`` to each
                item.

        Raises:
            ValueError: If ``split`` is not one of the manifest splits, or
                ``load_xvector`` is set without ``xvector_dir``.
            FileNotFoundError: If the AN4 archive, a manifest or
                ``xvector_dir`` is missing.
            RuntimeError: If ``manifest_path`` holds no utterance.

        Example:
            .. code-block:: python

                >>> valid = MiniAn4GANTTSDataset(
                ...     split="valid",
                ...     manifest_path="data/manifests_filtered/valid.tsv",
                ...     xvector_dir="data/x_vectors/voxcelebs12_rawnet3_valid",
                ...     load_speech=False,
                ...     inference=True,
                ... )
                >>> sorted(valid[0])
                ['raw_text', 'spembs', 'text', 'utt_id', 'wav_path']
        """
        if recipe_dir is None:
            recipe_dir = Path(__file__).resolve().parents[1]
        recipe_root = Path(recipe_dir).resolve()
        super().__init__(split, recipe_root)
        # The parent reads three columns; these manifests have a fourth, the
        # speaker id, so the rows are always read again here.
        if manifest_path is None:
            resolved_manifest = resolve_manifest_path(recipe_root, split)
        else:
            resolved_manifest = Path(manifest_path)
            if not resolved_manifest.is_absolute():
                resolved_manifest = recipe_root / resolved_manifest
        self._entries = self._read_entries(resolved_manifest)
        self.load_speech = load_speech
        self.load_xvector = load_xvector
        self.inference = inference
        self.xvector_dir = None
        if load_xvector:
            if xvector_dir is None:
                raise ValueError(
                    "xvector_dir must be supplied when load_xvector is True."
                )
            self.xvector_dir = Path(xvector_dir)
            if not self.xvector_dir.is_absolute():
                self.xvector_dir = recipe_root / self.xvector_dir
            if not self.xvector_dir.is_dir():
                raise FileNotFoundError(
                    f"x-vector directory not found: {self.xvector_dir}. "
                    "Run the compute_xvectors stage first."
                )

        fields = [Field("text", "text")]
        if load_speech:
            fields.append(Field("speech", "audio"))
        if load_xvector:
            fields.append(Field("spembs", "vector"))
        if inference:
            fields.extend(
                (
                    Field("utt_id", "text"),
                    Field("wav_path", "text"),
                    Field("raw_text", "text"),
                )
            )
        self.fields = tuple(fields)

    @staticmethod
    def _read_entries(manifest_path: Path) -> list[ManifestEntry]:
        """Read ``utt_id<TAB>wav_path<TAB>text`` rows, ignoring the speaker id."""
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        entries = []
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            for line in manifest_file:
                if not line.strip():
                    continue
                utt_id, wav_path, text = line.rstrip("\n").split("\t")[:3]
                entries.append(ManifestEntry(utt_id, Path(wav_path), text))
        if not entries:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")
        return entries

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one item with the fields declared in ``self.fields``.

        Args:
            idx: Positional index of the utterance.

        Returns:
            Dict with the transcript under ``text``; the waveform under
            ``speech`` as a 1-D float32 numpy array when ``load_speech``; the
            speaker embedding under ``spembs`` as a 1-D float32 numpy array
            when ``load_xvector``; and ``utt_id``, ``wav_path`` and
            ``raw_text`` as plain strings in ``inference`` mode.

        Raises:
            IndexError: If ``idx`` is out of range.
            FileNotFoundError: If the utterance has no embedding file.

        Example:
            .. code-block:: python

                >>> xvector_dir = "data/x_vectors/voxcelebs12_rawnet3_train"
                >>> dataset = MiniAn4GANTTSDataset(
                ...     split="train", xvector_dir=xvector_dir
                ... )
                >>> dataset[0]["spembs"].shape
                (192,)
        """
        entry = self._entries[int(idx)]
        sample: dict[str, Any] = {"text": entry.text}
        if self.load_speech:
            array, _sr = sf.read(str(entry.wav_path))
            sample["speech"] = np.asarray(array, dtype=np.float32)
        if self.load_xvector:
            pt_path = self.xvector_dir / f"{entry.utt_id}.pt"
            if not pt_path.is_file():
                raise FileNotFoundError(f"x-vector not found: {pt_path}")
            embedding = torch.load(pt_path, map_location="cpu")
            sample["spembs"] = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if self.inference:
            sample["utt_id"] = str(entry.utt_id)
            sample["wav_path"] = str(entry.wav_path)
            sample["raw_text"] = entry.text
        return sample
