"""Runner for parallel x-vector extraction."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from espnet3.parallel.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class XVectorRunner(BaseRunner):
    """Runner for extracting x-vectors (speaker embeddings) in parallel.

    Each utterance is saved as ``xvector_dir/{utt_id}.pt``, a 1-D tensor,
    immediately after extraction: if a target ``.pt`` already exists, the
    utterance is skipped without re-loading audio. The status of every
    utterance is appended to the shard-local ``results.jsonl`` the base
    runner's shard protocol writes, and :meth:`merge` reads those back in
    manifest order.

    Examples:
        Driven by the ``compute_xvectors`` stage through
        :class:`~espnet3.systems.esp2_gan_tts.src.xvector_provider.XVectorProvider`:
        ```python
        # `env` already carries xvector_dir, the model and the manifest.
        env = XVectorProvider(training_config, params=params).build_env_local()
        XVectorRunner.forward(range(10), **env)
        ```
    """

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        model: Any,
        toolkit: str,
        device: str,
        utterances: list,
        speaker_to_utterances: Dict[str, list],
        xvector_dir: Path,
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
            xvector_dir: Directory where per-utterance .pt files are written.
            config: Configuration object.
            **env: Additional environment entries.

        Returns:
            A status dict for an int index, or a list of status dicts for an
            iterable. Each entry is
            ``{"idx": int, "utt_id": str, "status": "ok"|"skipped"}``.

        Examples:
            ```python
            XVectorRunner.forward(0, **env)
            # -> {'utt_id': '19_198_000000_000000', 'status': 'ok'}

            # Re-running is a no-op for utterances already written:
            XVectorRunner.forward(0, **env)
            # -> {'utt_id': '19_198_000000_000000', 'status': 'skipped'}
            ```
        """
        if isinstance(idx, int):
            return XVectorRunner._process_one(
                idx, model, toolkit, device, utterances, xvector_dir
            )
        return [
            XVectorRunner._process_one(
                i, model, toolkit, device, utterances, xvector_dir
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
        xvector_dir: Path,
    ) -> Dict[str, Any]:
        """Extract one utterance's embedding and write it, skipping existing .pt."""
        utt_id, wav_path = utterances[idx]
        out_path = Path(xvector_dir) / f"{utt_id}.pt"
        if out_path.exists():
            return {"idx": idx, "utt_id": utt_id, "status": "skipped"}

        wav, in_sr = XVectorRunner._load_audio(wav_path)
        embedding = XVectorRunner._extract_embedding(wav, in_sr, model, toolkit, device)

        if isinstance(embedding, np.ndarray):
            tensor = torch.from_numpy(embedding).float()
        elif isinstance(embedding, torch.Tensor):
            tensor = embedding.float()
        else:
            tensor = torch.tensor(embedding, dtype=torch.float32)

        # One vector per utterance, whatever batch axis the toolkit returned
        # (espnet's Speech2Embedding gives (1, D)): the dataset serves it as
        # the 1-D `spembs` the inference contract declares.
        torch.save(tensor.reshape(-1), str(out_path))
        return {"idx": idx, "utt_id": utt_id, "status": "ok"}

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open the shard-local JSONL results file.

        Args:
            shard_dir: Directory of the shard being processed.
            **env: Environment entries, unused.

        Returns:
            ``{"results": file}``, where ``file`` is ``shard_dir/results.jsonl``
            opened for writing, so a shard that runs again starts empty.

        Examples:
            >>> writers = XVectorRunner.open_writers(Path("x_vectors/shards/split.0"))
            >>> sorted(writers)
            ['results']
        """
        results_path = Path(shard_dir) / "results.jsonl"
        return {"results": results_path.open("w", encoding="utf-8")}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Append one ``forward`` result (or batch of results) to the shard file.

        Args:
            writers: The dict returned by :meth:`open_writers`.
            result: A status dict, or a list of them for a batched call.
            state: Per-shard mutable state, unused.
            **env: Environment entries, unused.

        Examples:
            >>> XVectorRunner.write_record(
            ...     writers, {"idx": 0, "utt_id": "u1", "status": "ok"}, state={}
            ... )
        """
        records = result if isinstance(result, list) else [result]
        for record in records:
            writers["results"].write(json.dumps(record) + "\n")

    def merge(self, shard_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Concatenate shard results and restore manifest (``idx``) order.

        Args:
            shard_dirs: Directories of every shard of the run.

        Returns:
            All status dicts, sorted by ``idx``. A shard directory without a
            ``results.jsonl`` is skipped.

        Examples:
            >>> runner.merge([Path("x_vectors/shards/split.0")])
            [{'idx': 0, 'utt_id': 'u1', 'status': 'ok'}]
        """
        records: List[Dict[str, Any]] = []
        for shard_dir in shard_dirs:
            results_path = Path(shard_dir) / "results.jsonl"
            if not results_path.exists():
                continue
            with results_path.open("r", encoding="utf-8") as results_file:
                for line in results_file:
                    if line.strip():
                        records.append(json.loads(line))
        records.sort(key=lambda record: record["idx"])
        return records

    @staticmethod
    def _load_audio(wav_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Read an audio file as mono float32 at its native sample rate.

        soundfile returns ``(n_samples, n_channels)`` for multi-channel input,
        so anything with more than one channel is mixed down to a 1-D signal.
        """
        wav, in_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, in_sr

    @staticmethod
    def _resample(wav: np.ndarray, in_sr: int, tgt_sr: int) -> np.ndarray:
        """Resample along the last (time) axis, preserving the input shape."""
        tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        return torchaudio.functional.resample(tensor, in_sr, tgt_sr).numpy()

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
            wav = XVectorRunner._resample(wav, in_sr, tgt_sr)

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
        """Extract embedding using speechbrain toolkit.

        ``encode_batch`` expects ``(batch, time)`` and returns
        ``(batch, 1, emb)``; the single embedding is flattened to 1-D so that
        every toolkit persists the same ``(emb,)`` shape.
        """
        from speechbrain.dataio.preprocess import AudioNormalizer

        audio_norm = AudioNormalizer()
        wav_tensor = audio_norm(torch.from_numpy(wav), in_sr).unsqueeze(0).to(device)

        with torch.no_grad():
            embeddings = model.encode_batch(wav_tensor)

        return embeddings.detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _extract_rawnet(
        wav: np.ndarray, in_sr: int, model: Any, device: str
    ) -> np.ndarray:
        """Extract embedding using RawNet3 toolkit."""
        tgt_sr = 16000
        n_samples = 48000
        n_segments = 10

        if in_sr != tgt_sr:
            wav = XVectorRunner._resample(wav, in_sr, tgt_sr)

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
