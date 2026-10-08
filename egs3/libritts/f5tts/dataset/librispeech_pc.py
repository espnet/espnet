r"""LibriSpeech-PC cross-sentence eval set: manifest builder and dataset.

The F5-TTS repo's ``librispeech_pc_test_clean_cross_sentence.lst`` has six
tab-separated columns (ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt),
one same-speaker prompt/target pair per line. :func:`build_manifest` joins it
with the read-only LibriSpeech ``test-clean`` tree into the recipe-side
manifest

    gen_utt \t gen_text \t ref_utt \t ref_wav_path \t ref_text

which :class:`LibriSpeechPCDataset` serves. Every row pins its prompt,
reproducing the paper's fixed pairing. The output keys are the inputs
``espnet3.systems.f5tts.inference.Inference`` declares (``text``,
``reference_speech``, ``reference_text``), ``utt_id``, and the two scoring
references ``conf/metrics.yaml`` reads as ``dataset:<column>``
(``ref_wav_path`` for speaker similarity, ``text`` for WER).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset


def _find_utterance_flac(root: Path, utterance_id: str) -> Path:
    """Return the ``test-clean`` flac of a LibriSpeech utterance id."""
    speaker, chapter, _ = utterance_id.split("-")
    path = root / speaker / chapter / f"{utterance_id}.flac"
    if not path.exists():
        raise FileNotFoundError(f"Missing LibriSpeech audio: {path}")
    return path


def _read_lst(lst_path: Path) -> list[tuple[str, str, str, str]]:
    """Return ``(ref_utt, ref_text, gen_utt, gen_text)`` per pair-list row."""
    rows = []
    with Path(lst_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ref_utt, _ref_dur, ref_text, gen_utt, _gen_dur, gen_text = line.split("\t")
            rows.append((ref_utt, ref_text, gen_utt, gen_text))
    return rows


def build_manifest(lst_path, test_clean_root, out_tsv) -> int:
    """Write the LibriSpeech-PC manifest and return the number of rows.

    Args:
        lst_path: The F5-TTS cross-sentence pair list.
        test_clean_root: The LibriSpeech ``test-clean`` directory.
        out_tsv: Destination manifest path.

    Returns:
        The number of rows written.

    Raises:
        FileNotFoundError: If a prompt or target flac named by the pair list
            is missing from ``test_clean_root``.

    Example:
        .. code-block:: python

            build_manifest(
                "downloads/librispeech_pc_test_clean_cross_sentence.lst",
                "downloads/LibriSpeech/test-clean",
                "data/librispeech_pc/manifest.tsv",
            )
            # -> 1127
    """
    root = Path(test_clean_root).resolve()
    output_path = Path(out_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_lst(lst_path)
    with output_path.open("w", encoding="utf-8") as f:
        for ref_utt, ref_text, gen_utt, gen_text in rows:
            ref_wav = _find_utterance_flac(root, ref_utt)
            _find_utterance_flac(
                root, gen_utt
            )  # fail fast if the target audio is absent
            f.write(f"{gen_utt}\t{gen_text}\t{ref_utt}\t{ref_wav}\t{ref_text}\n")
    return len(rows)


class LibriSpeechPCDataset(TorchDataset):
    """Serve the LibriSpeech-PC manifest rows as F5-TTS inference samples.

    Each sample carries the target text plus the pinned prompt: ``utt_id``,
    ``text``, ``reference_speech`` (float32 mono at ``fs``), ``reference_text``
    and ``ref_wav_path``.

    Args:
        manifest_path: TSV written by :func:`build_manifest`.
        fs: Sampling rate the prompt audio is resampled to; ``None`` keeps
            the file's own rate.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        fs: int | None = 24000,
    ) -> None:
        """Read the five-column manifest into memory."""
        self.fs = fs
        self.rows: list[tuple[str, str, str, str, str]] = []
        with Path(manifest_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                gen_utt, gen_text, ref_utt, ref_wav, ref_text = line.split("\t")
                self.rows.append((gen_utt, gen_text, ref_utt, ref_wav, ref_text))
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        """Return the number of prompt/target pairs."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Load the prompt audio of pair ``idx`` and return the inference sample."""
        gen_utt, gen_text, _ref_utt, ref_wav, ref_text = self.rows[idx]
        speech, sample_rate = sf.read(ref_wav, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if self.fs is not None and sample_rate != self.fs:
            speech = torchaudio.functional.resample(
                torch.from_numpy(speech), sample_rate, self.fs
            ).numpy()
        return {
            "utt_id": gen_utt,
            "text": gen_text,
            "reference_speech": np.asarray(speech, dtype=np.float32),
            "reference_text": ref_text,
            "ref_wav_path": str(ref_wav),
        }


# Alias consumed by
# espnet3.components.data.dataset_module.instantiate_dataset_reference, which
# does `getattr(module, "Dataset")` after `import_module(data_src)` when a
# test split's `data_src` is set to this module's dotted path
Dataset = LibriSpeechPCDataset
