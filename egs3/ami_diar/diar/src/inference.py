"""Inference pipeline for diarization with speaker embeddings and clustering.

Based on DiariZen and pyannote.audio inference approach.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import medfilt

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    AgglomerativeClustering = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiarizationInference:
    """Inference pipeline for speaker diarization.

    Pipeline:
    1. Segmentation model: Get speaker activities (powerset-based)
    2. Optional median filtering: Smooth predictions
    3. Binarization: Convert probabilities to binary labels
    4. Speaker counting: Estimate number of speakers
    5. (Optional) Speaker embeddings: Extract embeddings for active speakers
    6. Clustering: Assign global speaker IDs
    7. Reconstruction: Generate final diarization output

    Args:
        model: Trained diarization model
        device: Device for inference ("cuda" or "cpu")
        apply_median_filtering: Whether to apply median filtering
        median_filter_size: Median filter kernel size (frames)
        binarization_threshold: Threshold for binarizing speaker activities
        use_speaker_embeddings: Whether to use speaker embeddings for clustering
        speaker_embedding_model: Path to speaker embedding model (ESPnet2)
        embedding_exclude_overlap: Exclude overlapping speech for embeddings
        clustering_backend: "ahc" (agglomerative) or "vbx" (variational bayes)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        apply_median_filtering: bool = True,
        median_filter_size: int = 11,
        binarization_threshold: float = 0.5,
        use_speaker_embeddings: bool = False,
        speaker_embedding_model: Optional[str] = None,
        embedding_exclude_overlap: bool = True,
        clustering_backend: str = "ahc",
        min_speakers: int = 1,
        max_speakers: int = 20,
        **clustering_kwargs,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.apply_median_filtering = apply_median_filtering
        self.median_filter_size = median_filter_size
        self.binarization_threshold = binarization_threshold

        self.use_speaker_embeddings = use_speaker_embeddings
        self.speaker_embedding_model_path = speaker_embedding_model
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.clustering_backend = clustering_backend
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.clustering_kwargs = clustering_kwargs

        # Load speaker embedding model if needed
        if self.use_speaker_embeddings:
            self._load_speaker_embedding_model()

    def _load_speaker_embedding_model(self):
        """Load ESPnet2 speaker embedding model."""
        if self.speaker_embedding_model_path is None:
            raise ValueError(
                "speaker_embedding_model must be specified when use_speaker_embeddings=True"
            )

        logger.info(
            f"Loading speaker embedding model from {self.speaker_embedding_model_path}"
        )

        # Import espnet2 speaker model
        from espnet2.bin.spk_inference import Speech2Embedding

        self.speaker_embedding_model = Speech2Embedding.from_pretrained(
            model_file=self.speaker_embedding_model_path,
            device=self.device,
        )
        logger.info("Speaker embedding model loaded successfully")

    def get_segmentations(
        self,
        waveform: torch.Tensor,
        soft: bool = True,
    ) -> np.ndarray:
        """Get speaker segmentations from waveform.

        Args:
            waveform: Input waveform (1D tensor or 2D with batch dim)
            soft: If True, return soft probabilities; else return binary labels

        Returns:
            Segmentations: (num_frames, num_speakers) array
        """
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        # Forward pass
        with torch.no_grad():
            speaker_activities, _ = self.model.inference(waveform)

        # Convert to numpy
        segmentations = speaker_activities.cpu().numpy()[0]  # Remove batch dim

        if not soft:
            segmentations = (segmentations >= self.binarization_threshold).astype(
                np.float32
            )

        return segmentations

    def apply_median_filter(
        self,
        segmentations: np.ndarray,
        kernel_size: int = 11,
    ) -> np.ndarray:
        """Apply median filter to smooth segmentations.

        Args:
            segmentations: (num_frames, num_speakers) array
            kernel_size: Median filter kernel size

        Returns:
            Filtered segmentations
        """
        num_frames, num_speakers = segmentations.shape
        filtered = np.zeros_like(segmentations)

        for spk_idx in range(num_speakers):
            filtered[:, spk_idx] = medfilt(
                segmentations[:, spk_idx], kernel_size=kernel_size
            )

        return filtered

    def count_speakers(
        self,
        segmentations: np.ndarray,
    ) -> int:
        """Estimate number of speakers from segmentations.

        Args:
            segmentations: Binary segmentations (num_frames, num_speakers)

        Returns:
            Estimated number of speakers
        """
        # Count speakers with at least some activity
        speaker_activity = segmentations.sum(axis=0)  # Total frames per speaker
        active_speakers = (speaker_activity > 0).sum()

        # Clip to valid range
        active_speakers = np.clip(active_speakers, self.min_speakers, self.max_speakers)

        return int(active_speakers)

    def extract_speaker_embeddings(
        self,
        waveform: torch.Tensor,
        segmentations: np.ndarray,
        frame_shift: float = 0.02,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract speaker embeddings for each speaker.

        Args:
            waveform: Input waveform (1D tensor)
            segmentations: Binary segmentations (num_frames, num_speakers)
            frame_shift: Frame shift in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Embeddings: (num_speakers, embedding_dim) array
        """
        if not self.use_speaker_embeddings:
            raise ValueError(
                "use_speaker_embeddings must be True to extract embeddings"
            )

        num_frames, num_speakers = segmentations.shape
        embeddings_list = []

        # Convert frame indices to sample indices
        samples_per_frame = int(frame_shift * sample_rate)

        for spk_idx in range(num_speakers):
            # Get frames where this speaker is active
            active_frames = np.where(segmentations[:, spk_idx] > 0)[0]

            if len(active_frames) == 0:
                # No activity for this speaker
                embeddings_list.append(None)
                continue

            # Optionally exclude overlapping speech
            if self.embedding_exclude_overlap:
                # Keep only frames with single speaker
                single_speaker_frames = segmentations.sum(axis=1) == 1
                active_frames = active_frames[single_speaker_frames[active_frames]]

            if len(active_frames) == 0:
                # No non-overlapping segments
                embeddings_list.append(None)
                continue

            # Extract audio segments for this speaker
            segments = []
            for frame_idx in active_frames:
                start_sample = frame_idx * samples_per_frame
                end_sample = min(start_sample + samples_per_frame, len(waveform))
                segment = waveform[start_sample:end_sample]
                segments.append(segment)

            # Concatenate segments
            speaker_audio = torch.cat(segments)

            # Extract embedding
            with torch.no_grad():
                embedding = self.speaker_embedding_model(speaker_audio.unsqueeze(0))[0]

            embeddings_list.append(embedding.cpu().numpy())

        # Stack embeddings (use zeros for inactive speakers)
        if embeddings_list[0] is not None:
            embedding_dim = embeddings_list[0].shape[0]
        else:
            embedding_dim = 256  # Default

        embeddings = np.zeros((num_speakers, embedding_dim))
        for spk_idx, emb in enumerate(embeddings_list):
            if emb is not None:
                embeddings[spk_idx] = emb

        return embeddings

    def cluster_speakers(
        self,
        segmentations: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        num_speakers: Optional[int] = None,
    ) -> np.ndarray:
        """Cluster speakers to assign global speaker IDs.

        Args:
            segmentations: Binary segmentations (num_frames, num_speakers)
            embeddings: Speaker embeddings (num_speakers, embedding_dim)
            num_speakers: Number of speakers (if None, auto-detect)

        Returns:
            Cluster assignments: (num_speakers,) array
                Maps local speaker index to global cluster ID
        """
        if num_speakers is None:
            num_speakers = self.count_speakers(segmentations)

        num_local_speakers = segmentations.shape[1]

        # If not using embeddings, use co-occurrence statistics
        if embeddings is None or not self.use_speaker_embeddings:
            # Use speaker co-occurrence for clustering
            # For simplicity, just use identity mapping
            # (assumes local speakers are already globally consistent)
            return np.arange(num_local_speakers)

        # Cluster based on embeddings
        if self.clustering_backend == "ahc":
            return self._cluster_ahc(embeddings, num_speakers)
        elif self.clustering_backend == "vbx":
            return self._cluster_vbx(embeddings, num_speakers)
        else:
            raise ValueError(f"Unknown clustering backend: {self.clustering_backend}")

    def _cluster_ahc(
        self,
        embeddings: np.ndarray,
        num_speakers: int,
    ) -> np.ndarray:
        """Agglomerative hierarchical clustering.

        Args:
            embeddings: (num_speakers, embedding_dim)
            num_speakers: Target number of clusters

        Returns:
            Cluster assignments: (num_speakers,)
        """
        if AgglomerativeClustering is None:
            raise ImportError("scikit-learn is required for AHC clustering")

        metric = self.clustering_kwargs.get("metric", "cosine")
        linkage = self.clustering_kwargs.get("linkage", "average")

        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric=metric,
            linkage=linkage,
        )

        labels = clustering.fit_predict(embeddings)
        return labels

    def _cluster_vbx(
        self,
        embeddings: np.ndarray,
        num_speakers: int,
    ) -> np.ndarray:
        """Variational Bayes clustering (VBx).

        Note: This is a placeholder. For full VBx implementation,
        integrate with VBDiarization library or similar.

        Args:
            embeddings: (num_speakers, embedding_dim)
            num_speakers: Target number of clusters

        Returns:
            Cluster assignments: (num_speakers,)
        """
        raise NotImplementedError

    def reconstruct_diarization(
        self,
        segmentations: np.ndarray,
        cluster_labels: np.ndarray,
        frame_shift: float = 0.02,
    ) -> List[Tuple[float, float, int]]:
        """Reconstruct final diarization output.

        Args:
            segmentations: Binary segmentations (num_frames, num_speakers)
            cluster_labels: Cluster assignments (num_speakers,)
            frame_shift: Frame shift in seconds

        Returns:
            List of (start_time, end_time, speaker_id) tuples
        """
        num_frames, num_speakers = segmentations.shape
        diarization = []

        # Create global speaker activities
        # Map local speakers to global clusters
        num_clusters = cluster_labels.max() + 1
        global_activities = np.zeros((num_frames, num_clusters))

        for local_spk in range(num_speakers):
            global_spk = cluster_labels[local_spk]
            global_activities[:, global_spk] = np.maximum(
                global_activities[:, global_spk], segmentations[:, local_spk]
            )

        # Extract segments for each global speaker
        for global_spk in range(num_clusters):
            active_frames = np.where(global_activities[:, global_spk] > 0)[0]

            if len(active_frames) == 0:
                continue

            # Group consecutive frames into segments
            segments = []
            start_frame = active_frames[0]
            prev_frame = active_frames[0]

            for frame in active_frames[1:]:
                if frame > prev_frame + 1:
                    # Gap detected, end current segment
                    end_frame = prev_frame
                    segments.append((start_frame, end_frame))
                    start_frame = frame
                prev_frame = frame

            # Add last segment
            segments.append((start_frame, prev_frame))

            # Convert frame indices to time
            for start_frame, end_frame in segments:
                start_time = start_frame * frame_shift
                end_time = (end_frame + 1) * frame_shift
                diarization.append((start_time, end_time, global_spk))

        # Sort by start time
        diarization.sort(key=lambda x: x[0])

        return diarization

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float, int]]:
        """Run inference on waveform.

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate in Hz

        Returns:
            Diarization output: List of (start_time, end_time, speaker_id)
        """
        # 1. Get segmentations
        segmentations = self.get_segmentations(waveform, soft=True)

        # 2. Apply median filtering
        if self.apply_median_filtering:
            segmentations = self.apply_median_filter(
                segmentations, kernel_size=self.median_filter_size
            )

        # 3. Binarize
        binary_segmentations = (segmentations >= self.binarization_threshold).astype(
            np.float32
        )

        # 4. Count speakers
        num_speakers = self.count_speakers(binary_segmentations)
        logger.info(f"Estimated {num_speakers} speakers")

        # 5. Extract embeddings (if enabled)
        embeddings = None
        if self.use_speaker_embeddings:
            embeddings = self.extract_speaker_embeddings(
                waveform,
                binary_segmentations,
                sample_rate=sample_rate,
            )

        # 6. Cluster speakers
        cluster_labels = self.cluster_speakers(
            binary_segmentations,
            embeddings=embeddings,
            num_speakers=num_speakers,
        )

        # 7. Reconstruct diarization
        diarization = self.reconstruct_diarization(
            binary_segmentations,
            cluster_labels,
        )

        return diarization


def save_rttm(
    diarization: List[Tuple[float, float, int]],
    output_path: Path,
    recording_id: str,
):
    """Save diarization output in RTTM format.

    Args:
        diarization: List of (start_time, end_time, speaker_id)
        output_path: Output file path
        recording_id: Recording ID
    """
    with open(output_path, "w") as f:
        for start_time, end_time, speaker_id in diarization:
            duration = end_time - start_time
            # RTTM format:
            # SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
            f.write(
                f"SPEAKER {recording_id} 1 {start_time:.3f} {duration:.3f} "
                f"<NA> <NA> speaker_{speaker_id} <NA> <NA>\n"
            )


if __name__ == "__main__":
    # Test inference pipeline
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <audio_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_path = sys.argv[2]

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")

    # Load audio
    import torchaudio

    logger.info(f"Loading audio from {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]

    # Run inference
    inference = DiarizationInference(
        model=model,
        device="cpu",
        apply_median_filtering=True,
        use_speaker_embeddings=False,
    )

    diarization = inference(waveform, sample_rate=sample_rate)

    # Print results
    print("\nDiarization results:")
    for start, end, speaker in diarization:
        print(f"  {start:.2f} - {end:.2f}: Speaker {speaker}")

    # Save RTTM
    output_path = Path(audio_path).with_suffix(".rttm")
    recording_id = Path(audio_path).stem
    save_rttm(diarization, output_path, recording_id)
    print(f"\nSaved RTTM to {output_path}")
