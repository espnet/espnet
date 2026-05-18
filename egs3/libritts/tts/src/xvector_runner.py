"""Runner for parallel x-vector extraction."""

import logging
from typing import Any, Dict, Iterable, Optional, Union

import kaldiio
import librosa
import numpy as np
import torch
from pathlib import Path

from espnet3.parallel.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class XVectorRunner(BaseRunner):
    """Runner for extracting x-vectors (speaker embeddings) in parallel.
    
    This runner processes utterances to extract speaker embeddings and
    supports per-speaker averaging. Results are written to ark/scp format.
    """

    def __init__(self, provider, output_dir: str, spk_embed_tag: str = "spk_embed", **kwargs):
        """Initialize the x-vector runner.
        
        Args:
            provider: EnvironmentProvider instance
            output_dir: Directory to save results
            spk_embed_tag: Tag for embedding files (used in filename)
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(provider, **kwargs)
        self.output_dir = Path(output_dir)
        self.spk_embed_tag = spk_embed_tag
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        model: Any,
        toolkit: str,
        device: str,
        utterances: list,
        speaker_to_utterances: Dict[str, list],
        config: Any,
        **env
    ) -> Dict[str, Any]:
        """Extract x-vectors for given utterances from manifest.
        
        Args:
            idx: Single index or batch of indices into utterances list
            model: Speaker embedding model
            toolkit: Type of toolkit used ('espnet', 'speechbrain', 'rawnet')
            device: Device model is on
            utterances: List of (utt_id, wav_path) tuples from manifest
            speaker_to_utterances: Speaker-to-utterance mapping from manifest
            config: Configuration object
            **env: Additional environment entries
            
        Returns:
            Dictionary with extracted embeddings and metadata
        """
        # Handle batch vs single index
        if isinstance(idx, int):
            indices = [idx]
        else:
            indices = list(idx)
        
        results = []
        for idx_val in indices:
            try:
                # Get utterance from manifest
                utt_id, wav_path = utterances[idx_val]
                wav_path = Path(wav_path)
                
                if not wav_path.exists():
                    logger.warning(f"Audio file not found: {wav_path}")
                    continue
                
                # Load audio using librosa
                wav, in_sr = librosa.load(str(wav_path), sr=None)
                
                # Extract embedding
                embedding = XVectorRunner._extract_embedding(
                    wav, in_sr, model, toolkit, device
                )
                
                results.append({
                    "utt_id": utt_id,
                    "embedding": embedding,
                })
            except Exception as e:
                logger.error(f"Error processing index {idx_val}: {e}")
                continue
        
        return results

    @staticmethod
    def _extract_embedding(
        wav: np.ndarray,
        in_sr: int,
        model: Any,
        toolkit: str,
        device: str,
    ) -> np.ndarray:
        """Extract speaker embedding from audio.
        
        Args:
            wav: Audio waveform
            in_sr: Sample rate of input audio
            model: Speaker embedding model
            toolkit: Type of toolkit
            device: Device model is on
            
        Returns:
            Speaker embedding vector
        """
        if toolkit == "espnet":
            return XVectorRunner._extract_espnet(wav, in_sr, model, device)
        elif toolkit == "speechbrain":
            return XVectorRunner._extract_speechbrain(wav, in_sr, model, device)
        elif toolkit == "rawnet":
            return XVectorRunner._extract_rawnet(wav, in_sr, model, device)
        else:
            raise ValueError(f"Unknown toolkit: {toolkit}")

    @staticmethod
    def _extract_espnet(wav: np.ndarray, in_sr: int, model: Any, device: str) -> np.ndarray:
        """Extract embedding using espnet toolkit."""
        tgt_sr = 16000 # follow espnet2 default
        
        # Resample if necessary
        if in_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=tgt_sr)
        
        # Handle multi-channel
        if len(wav.shape) == 2:
            wav = np.mean(wav, axis=0)
        
        # Convert to tensor and extract
        wav_tensor = torch.from_numpy(wav.astype(np.float32)).to(device)
        with torch.no_grad():
            output = model(wav_tensor)
        
        return output.cpu().numpy()

    @staticmethod
    def _extract_speechbrain(wav: np.ndarray, in_sr: int, model: Any, device: str) -> np.ndarray:
        """Extract embedding using speechbrain toolkit."""
        from speechbrain.dataio.preprocess import AudioNormalizer
        
        audio_norm = AudioNormalizer()
        wav_tensor = audio_norm(torch.from_numpy(wav), in_sr).to(device)
        
        with torch.no_grad():
            embeddings = model.encode_batch(wav_tensor)
        
        return embeddings.detach().cpu().numpy()[0]

    @staticmethod
    def _extract_rawnet(wav: np.ndarray, in_sr: int, model: Any, device: str) -> np.ndarray:
        """Extract embedding using RawNet3 toolkit."""
        tgt_sr = 16000
        n_samples = 48000
        n_segments = 10
        
        # Resample if necessary
        if in_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=tgt_sr)
        
        # Pad if too short (RawNet3 trained on 3-second utterances)
        if len(wav) < n_samples:
            shortage = n_samples - len(wav) + 1
            wav = np.pad(wav, (0, shortage), "wrap")
        
        # Create multiple segments
        audios = []
        startframe = np.linspace(0, len(wav) - n_samples, num=n_segments)
        for asf in startframe:
            audios.append(wav[int(asf) : int(asf) + n_samples])
        
        audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32)).to(device)
        
        with torch.no_grad():
            output = model(audios)
        
        # Average across segments
        return output.mean(0).detach().cpu().numpy()

    def write_results(self, results: list, suffix: str = "") -> None:
        """Write extracted embeddings to separate PyTorch files.
        
        Saves each embedding as a separate .pt file for memory efficiency.
        
        File structure: output_dir/{spk_embed_tag}{suffix}/{utt_id}.pt
        
        Args:
            results: List of results from forward passes
            suffix: Suffix to add to output filenames
        
        Raises:
            RuntimeError: If results are empty or invalid.
        """
        if not results:
            raise RuntimeError("No results to write. Ensure that forward() returns valid results before calling write_results().")
        
        # Flatten results from parallel execution
        all_results = []
        for batch_results in results:
            if isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)
        
        if not all_results:
            raise RuntimeError("No results to write. Ensure that forward() returns valid results before calling write_results().")
        
        # Prepare output base path
        output_tag = f"{self.spk_embed_tag}{suffix}"
        (self.output_dir / output_tag).mkdir(parents=True, exist_ok=True)
        
        # Write each embedding to a separate file
        logger.info(f"Writing utterance embeddings to {self.output_dir / output_tag}")
        for result in all_results:
            utt_id = result["utt_id"]
            embedding = result["embedding"]
            
            # Convert to torch tensor if not already
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding).float()
            elif not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            # Write to individual file
            pt_path = self.output_dir / f"{output_tag}" / f"{utt_id}.pt"
            torch.save(embedding, str(pt_path))
        
        logger.info(f"Wrote {len(all_results)} utterance embeddings to {self.output_dir / output_tag}")
