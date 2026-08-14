"""Runner for parallel x-vector extraction."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import librosa
import numpy as np
import torch

from espnet3.parallel.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class XVectorRunner(BaseRunner):
    """Runner for extracting x-vectors (speaker embeddings) in parallel.

    Each utterance is saved as ``output_dir/{utt_id}.pt`` immediately after
    extraction: if a target ``.pt`` already exists, the utterance is skipped
    without re-loading audio.
    """

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        model: Any,
        toolkit: str,
        device: str,
        utterances: list,
        speaker_to_utterances: Dict[str, list],
        output_dir: Path,
        config: Any,
        **env,
    ) -> Union[Dict[str, Any], list]:
        """Extract and persist x-vectors for the given index or batch.

        Args:
            idx: Single index or iterable of indices into ``utterances``.
            model: Speaker embedding model.
            toolkit: 'espnet', 'speechbrain', or 'rawnet'.
            device: Device the model lives on.
            utterances: List of (utt_id, wav_path) tuples from manifest.
            speaker_to_utterances: Speaker-to-utterance mapping from manifest.
            output_dir: Directory where per-utterance .pt files are written.
            config: Configuration object.
            **env: Additional environment entries.

        Returns:
            A status dict for an int index, or a list of status dicts for an
            iterable. Each entry is ``{"utt_id": str, "status": "ok"|"skipped"}``.
        """
        if isinstance(idx, int):
            return XVectorRunner._process_one(
                idx, model, toolkit, device, utterances, output_dir
            )
        return [
            XVectorRunner._process_one(
                i, model, toolkit, device, utterances, output_dir
            )
            for i in idx
        ]

    @staticmethod
    def _process_one(
        idx: int,
        model: Any,
        toolkit: str,
        device: str,
        utterances: list,
        output_dir: Path,
    ) -> Dict[str, Any]:
        utt_id, wav_path = utterances[idx]
        out_path = Path(output_dir) / f"{utt_id}.pt"
        if out_path.exists():
            return {"utt_id": utt_id, "status": "skipped"}

        wav, in_sr = librosa.load(str(wav_path), sr=None)
        embedding = XVectorRunner._extract_embedding(wav, in_sr, model, toolkit, device)

        if isinstance(embedding, np.ndarray):
            tensor = torch.from_numpy(embedding).float()
        elif isinstance(embedding, torch.Tensor):
            tensor = embedding.float()
        else:
            tensor = torch.tensor(embedding, dtype=torch.float32)

        torch.save(tensor, str(out_path))
        return {"utt_id": utt_id, "status": "ok"}

    @staticmethod
    def _extract_embedding(
        wav: np.ndarray,
        in_sr: int,
        model: Any,
        toolkit: str,
        device: str,
    ) -> np.ndarray:
        """Extract speaker embedding from audio."""
        if toolkit == "espnet":
            return XVectorRunner._extract_espnet(wav, in_sr, model, device)
        elif toolkit == "speechbrain":
            return XVectorRunner._extract_speechbrain(wav, in_sr, model, device)
        elif toolkit == "rawnet":
            return XVectorRunner._extract_rawnet(wav, in_sr, model, device)
        else:
            raise ValueError(f"Unknown toolkit: {toolkit}")

    @staticmethod
    def _extract_espnet(
        wav: np.ndarray, in_sr: int, model: Any, device: str
    ) -> np.ndarray:
        """Extract embedding using espnet toolkit."""
        tgt_sr = 16000  # follow espnet2 default

        if in_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=tgt_sr)

        if len(wav.shape) == 2:
            wav = np.mean(wav, axis=0)

        wav_tensor = torch.from_numpy(wav.astype(np.float32)).to(device)
        with torch.no_grad():
            output = model(wav_tensor)

        return output.cpu().numpy()

    @staticmethod
    def _extract_speechbrain(
        wav: np.ndarray, in_sr: int, model: Any, device: str
    ) -> np.ndarray:
        """Extract embedding using speechbrain toolkit."""
        from speechbrain.dataio.preprocess import AudioNormalizer

        audio_norm = AudioNormalizer()
        wav_tensor = audio_norm(torch.from_numpy(wav), in_sr).to(device)

        with torch.no_grad():
            embeddings = model.encode_batch(wav_tensor)

        return embeddings.detach().cpu().numpy()[0]

    @staticmethod
    def _extract_rawnet(
        wav: np.ndarray, in_sr: int, model: Any, device: str
    ) -> np.ndarray:
        """Extract embedding using RawNet3 toolkit."""
        tgt_sr = 16000
        n_samples = 48000
        n_segments = 10

        if in_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=tgt_sr)

        # RawNet3 was trained on 3-second utterances; pad shorter clips.
        if len(wav) < n_samples:
            shortage = n_samples - len(wav) + 1
            wav = np.pad(wav, (0, shortage), "wrap")

        audios = []
        startframe = np.linspace(0, len(wav) - n_samples, num=n_segments)
        for asf in startframe:
            audios.append(wav[int(asf) : int(asf) + n_samples])

        audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32)).to(
            device
        )

        with torch.no_grad():
            output = model(audios)

        return output.mean(0).detach().cpu().numpy()
