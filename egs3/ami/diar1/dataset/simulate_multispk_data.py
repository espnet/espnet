import random
import numpy as np
import torch
import torchaudio
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import lhotse
from lhotse.manipulation import combine as combine_manifests
from lhotse import CutSet, SupervisionSegment, Recording, MonoCut, AudioSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIBRISPLITs = ["train-clean-100", "train-clean-360", "train-other-500"]


class TransitionType(Enum):
    """Four types of utterance transitions as defined in the paper"""
    TURN_HOLD = "TH"  # Same speaker with pause
    TURN_SWITCH = "TS"  # Different speaker with gap
    INTERRUPTION = "IR"  # Different speaker with overlap
    BACKCHANNEL = "BC"  # Different speaker fully overlapped


@dataclass
class TransitionParams:
    """Parameters for each transition type"""
    beta_th: float = 0.57  # Expected pause duration for turn-hold
    beta_ts: float = 0.40  # Expected gap duration for turn-switch
    beta_ir: float = 0.10  # Expected overlap ratio for interruption
    beta_bc: float = 0.44  # Expected overlap ratio for backchannel

    # Probability distributions
    p_ind: List[float] = None  # [p_TH, p_TS, p_IR, p_BC] for random selection
    p_markov: np.ndarray = None  # 4x4 transition matrix for Markov selection

    def __post_init__(self):
        if self.p_ind is None:
            # Default probabilities from CALLHOME1 (Table 1 in paper)
            self.p_ind = [0.15, 0.31, 0.44, 0.10]

        if self.p_markov is None:
            # Default Markov transition matrix from CALLHOME1
            self.p_markov = np.array([
                [0.26, 0.11, 0.09, 0.31],  # TH -> [TH, TS, IR, BC]
                [0.23, 0.38, 0.29, 0.29],  # TS -> [TH, TS, IR, BC]
                [0.27, 0.45, 0.53, 0.31],  # IR -> [TH, TS, IR, BC]
                [0.24, 0.06, 0.09, 0.09]   # BC -> [TH, TS, IR, BC]
            ])


class ConversationSimulator:
    """
    Multi-speaker conversation simulator based on the paper:
    'Improving the Naturalness of Simulated Conversations for End-to-End Neural Diarization'
    """

    def __init__(self,
                 transition_params: TransitionParams = None,
                 sample_rate: int = 16000,
                 use_markov: bool = True):
        """
        Initialize the conversation simulator

        Args:
            transition_params: Parameters for transition types
            sample_rate: Audio sample rate
            use_markov: Whether to use Markov chain for transition selection
        """
        self.params = transition_params or TransitionParams()
        self.sample_rate = sample_rate
        self.use_markov = use_markov
        self.epsilon = 0.03  # For truncated exponential distribution

        # Validate Markov matrix
        if self.use_markov:
            row_sums = np.sum(self.params.p_markov, axis=1)
            if not np.allclose(row_sums, 1.0):
                logger.warning("Markov matrix rows don't sum to 1, normalizing...")
                self.params.p_markov = self.params.p_markov / row_sums[:, np.newaxis]

    def sample_exponential_duration(self, beta: float) -> float:
        """Sample duration from exponential distribution"""
        return np.random.exponential(beta)

    def sample_overlap_ratio(self, beta: float) -> float:
        """Sample overlap ratio from truncated exponential distribution"""
        # Sample from exponential and truncate to [epsilon, 1-epsilon]
        ratio = np.random.exponential(beta)
        return np.clip(ratio, self.epsilon, 1.0 - self.epsilon)

    def select_transition_type(self, prev_transition: Optional[TransitionType] = None) -> TransitionType:
        """Select next transition type based on random or Markov selection"""
        if self.use_markov and prev_transition is not None:
            # Use Markov chain
            prev_idx = list(TransitionType).index(prev_transition)
            probs = self.params.p_markov[prev_idx]
        else:
            # Use independent random selection
            probs = self.params.p_ind

        # Sample transition type
        transition_idx = np.random.choice(len(TransitionType), p=probs)
        return list(TransitionType)[transition_idx]

    def apply_transition(self,
                         prev_audio: torch.Tensor,
                         next_audio: torch.Tensor,
                         transition_type: TransitionType,
                         prev_end_time: float) -> Tuple[torch.Tensor, float, float]:
        """
        Apply transition between two audio segments

        Returns:
            combined_audio: Combined audio with transition applied
            next_start_time: Start time of next utterance
            next_end_time: End time of next utterance
        """
        prev_duration = len(prev_audio) / self.sample_rate
        next_duration = len(next_audio) / self.sample_rate

        if transition_type == TransitionType.TURN_HOLD:
            # Same speaker with pause
            pause_duration = self.sample_exponential_duration(self.params.beta_th)
            pause_samples = int(pause_duration * self.sample_rate)
            pause_audio = torch.zeros(pause_samples)

            combined_audio = torch.cat([prev_audio, pause_audio, next_audio])
            next_start_time = prev_end_time + pause_duration
            next_end_time = next_start_time + next_duration

        elif transition_type == TransitionType.TURN_SWITCH:
            # Different speaker with gap
            gap_duration = self.sample_exponential_duration(self.params.beta_ts)
            gap_samples = int(gap_duration * self.sample_rate)
            gap_audio = torch.zeros(gap_samples)

            combined_audio = torch.cat([prev_audio, gap_audio, next_audio])
            next_start_time = prev_end_time + gap_duration
            next_end_time = next_start_time + next_duration

        elif transition_type == TransitionType.INTERRUPTION:
            # Different speaker with partial overlap
            overlap_ratio = self.sample_overlap_ratio(self.params.beta_ir)
            overlap_duration = overlap_ratio * min(prev_duration, next_duration)
            overlap_samples = int(overlap_duration * self.sample_rate)

            # Ensure we don't exceed audio lengths
            overlap_samples = min(overlap_samples, len(prev_audio), len(next_audio))

            # Mix overlapping portions
            prev_non_overlap = prev_audio[:-overlap_samples] if overlap_samples > 0 else prev_audio
            overlap_prev = prev_audio[-overlap_samples:] if overlap_samples > 0 else torch.zeros(0)
            overlap_next = next_audio[:overlap_samples] if overlap_samples > 0 else torch.zeros(0)
            next_non_overlap = next_audio[overlap_samples:] if overlap_samples > 0 else next_audio

            # Mix overlapping parts
            if len(overlap_prev) > 0 and len(overlap_next) > 0:
                mixed_overlap = overlap_prev + overlap_next
            else:
                mixed_overlap = torch.zeros(0)

            combined_audio = torch.cat([prev_non_overlap, mixed_overlap, next_non_overlap])
            next_start_time = prev_end_time - overlap_duration
            next_end_time = next_start_time + next_duration

        elif transition_type == TransitionType.BACKCHANNEL:
            # Different speaker fully overlapped (backchannel)
            overlap_ratio = self.sample_overlap_ratio(self.params.beta_bc)

            # Determine overlap duration and start position
            shorter_duration = min(prev_duration, next_duration)
            overlap_duration = overlap_ratio * shorter_duration

            # Randomly place the shorter utterance within the longer one
            if next_duration <= prev_duration:
                # Next utterance is shorter, place it randomly in prev
                max_start_offset = prev_duration - next_duration
                start_offset = np.random.uniform(0, max_start_offset)
                start_samples = int(start_offset * self.sample_rate)

                # Create output audio same length as prev_audio
                combined_audio = prev_audio.clone()
                end_samples = start_samples + len(next_audio)
                end_samples = min(end_samples, len(combined_audio))

                # Mix the overlapping portion
                combined_audio[start_samples:end_samples] += next_audio[:end_samples - start_samples]

                next_start_time = prev_end_time - prev_duration + start_offset
                next_end_time = next_start_time + next_duration
            else:
                # Prev utterance is shorter, extend combined audio
                max_start_offset = next_duration - prev_duration
                start_offset = np.random.uniform(0, max_start_offset)
                start_samples = int(start_offset * self.sample_rate)

                # Create combined audio
                total_samples = len(prev_audio) + start_samples + len(next_audio)
                combined_audio = torch.zeros(total_samples)
                combined_audio[:len(prev_audio)] = prev_audio
                combined_audio[start_samples:start_samples + len(next_audio)] += next_audio

                next_start_time = prev_end_time - prev_duration + start_offset
                next_end_time = next_start_time + next_duration

        return combined_audio, next_start_time, next_end_time

    def simulate_conversation(self,
                              speaker_cuts: Dict[str, List[MonoCut]],
                              num_utterances: int = 50,
                              target_speakers: List[str] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Simulate a multi-speaker conversation

        Args:
            speaker_cuts: Dictionary mapping speaker IDs to their cuts
            num_utterances: Total number of utterances in conversation
            target_speakers: List of speakers to include (if None, use all)

        Returns:
            conversation_audio: Combined conversation audio
            supervision_segments: List of supervision segments with timing
        """
        if target_speakers is None:
            target_speakers = list(speaker_cuts.keys())

        # Ensure we have enough cuts for each speaker
        for speaker in target_speakers:
            if len(speaker_cuts[speaker]) == 0:
                raise ValueError(f"No cuts available for speaker {speaker}")

        # Initialize conversation
        conversation_audio = torch.tensor([])
        supervision_segments = []
        current_time = 0.0
        prev_transition = None
        prev_speaker = None

        for utt_idx in range(num_utterances):
            # Select transition type
            transition_type = self.select_transition_type(prev_transition)

            # Select speaker based on transition type
            if transition_type == TransitionType.TURN_HOLD:
                # Same speaker
                if prev_speaker is not None:
                    current_speaker = prev_speaker
                else:
                    current_speaker = np.random.choice(target_speakers)
            else:
                # Different speaker
                if prev_speaker is not None:
                    available_speakers = [s for s in target_speakers if s != prev_speaker]
                    current_speaker = np.random.choice(available_speakers)
                else:
                    current_speaker = np.random.choice(target_speakers)

            # Select random cut for current speaker
            cut = np.random.choice(speaker_cuts[current_speaker])

            # Load audio
            audio = torch.from_numpy(cut.load_audio()).float()
            if len(audio.shape) > 1:
                audio = audio.mean(dim=0)  # Convert to mono if needed

            # Resample if necessary
            if cut.sampling_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=cut.sampling_rate,
                    new_freq=self.sample_rate
                )
                audio = resampler(audio)

            if utt_idx == 0:
                # First utterance
                conversation_audio = audio
                start_time = 0.0
                end_time = len(audio) / self.sample_rate
            else:
                # Apply transition
                combined_audio, start_time, end_time = self.apply_transition(
                    conversation_audio, audio, transition_type, current_time
                )
                conversation_audio = combined_audio

            # Update supervision segments
            supervision_segments.append({
                'speaker': current_speaker,
                'start': start_time,
                'duration': end_time - start_time,
                'text': getattr(cut, 'text', ''),
                'transition_type': transition_type.value
            })

            # Update state
            current_time = end_time
            prev_transition = transition_type
            prev_speaker = current_speaker

            logger.debug(f"Utterance {utt_idx}: {current_speaker}, "
                         f"transition: {transition_type.value}, "
                         f"time: {start_time:.2f}-{end_time:.2f}s")

        return conversation_audio, supervision_segments


def create_librispeech_conversation_dataset(librispeech_cuts: CutSet,
                                            num_conversations: int = 100,
                                            num_utterances_per_conversation: int = 50,
                                            num_speakers_per_conversation: int = 2,
                                            output_dir: Path = None,
                                            min_utterance_duration: float = 0.1,
                                            max_utterance_duration: float = 60.0) -> CutSet:
    """
    Create a simulated conversation dataset from LibriSpeech

    Args:
        librispeech_cuts: LibriSpeech CutSet
        num_conversations: Number of conversations to generate
        num_utterances_per_conversation: Utterances per conversation
        num_speakers_per_conversation: Speakers per conversation
        output_dir: Directory to save generated conversations
        min_utterance_duration: Minimum utterance duration
        max_utterance_duration: Maximum utterance duration

    Returns:
        CutSet containing simulated conversations
    """
    # Filter cuts by duration
    filtered_cuts = librispeech_cuts.filter(
        lambda cut: min_utterance_duration <= cut.duration <= max_utterance_duration
    )

    # Group cuts by speaker
    speaker_cuts = {}
    for cut in filtered_cuts:
        speaker_id = cut.supervisions[0].speaker
        if speaker_id not in speaker_cuts:
            speaker_cuts[speaker_id] = []
        speaker_cuts[speaker_id].append(cut)

    # Filter speakers with enough utterances
    min_utterances = max(10, num_utterances_per_conversation // num_speakers_per_conversation)
    valid_speakers = {
        speaker: cuts for speaker, cuts in speaker_cuts.items()
        if len(cuts) >= min_utterances
    }

    if len(valid_speakers) < num_speakers_per_conversation:
        raise ValueError(f"Not enough speakers with sufficient utterances. "
                         f"Found {len(valid_speakers)}, need {num_speakers_per_conversation}")

    logger.info(f"Found {len(valid_speakers)} valid speakers")

    # Initialize simulator
    simulator = ConversationSimulator(use_markov=True)

    # Generate conversations
    conversation_cuts = []

    for conv_idx in range(num_conversations):
        logger.info(f"Generating conversation {conv_idx + 1}/{num_conversations}")

        # Select random speakers for this conversation
        selected_speakers = np.random.choice(
            list(valid_speakers.keys()),
            size=num_speakers_per_conversation,
            replace=False
        )

        # Create speaker cuts subset
        conv_speaker_cuts = {
            speaker: valid_speakers[speaker]
            for speaker in selected_speakers
        }

        # Simulate conversation
        #try:
        conv_audio, supervision_segments = simulator.simulate_conversation(
            conv_speaker_cuts,
            num_utterances=num_utterances_per_conversation,
            target_speakers=list(selected_speakers)
        )

        # Create recording
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / f"conversation_{conv_idx:08d}.wav"
            torchaudio.save(
                audio_path,
                conv_audio.unsqueeze(0),
                sample_rate=simulator.sample_rate
            )
        else:
            audio_path = f"conversation_{conv_idx:08d}"

        recording = Recording(
            id=f"conv_{conv_idx:08d}",
            sources=[AudioSource(type='file',
                                 channels=[0],
                                 source=str(audio_path))],
            sampling_rate=simulator.sample_rate,
            num_samples=len(conv_audio),
            duration=len(conv_audio) / simulator.sample_rate
        )

        # Create supervisions
        supervisions = []
        for seg_idx, seg in enumerate(supervision_segments):
            supervision = SupervisionSegment(
                id=f"conv_{conv_idx:08d}_{seg_idx:08d}",
                recording_id=recording.id,
                start=seg['start'],
                duration=seg['duration'],
                speaker=seg['speaker'],
                text=seg['text'],
                custom={'transition_type': seg['transition_type']})
            supervisions.append(supervision)

        # Create cut
        cut = MonoCut(
            id=f"conv_{conv_idx:08d}",
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=supervisions
        )

        conversation_cuts.append(cut)

        #except Exception as e:
        #    logger.error(f"Failed to generate conversation {conv_idx}: {e}")
        #    continue

    logger.info(f"Successfully generated {len(conversation_cuts)} conversations")
    return CutSet.from_cuts(conversation_cuts)


# Example usage
if __name__ == "__main__":
    # This is an example of how to use the simulator
    # You would need to load your LibriSpeech dataset first

    lhotse_manifest_dir = "./librispeech_manifests"
    # Example: Load LibriSpeech cuts (you need to create this)
    from lhotse.recipes.librispeech import prepare_librispeech
    prepare_librispeech(
         corpus_dir=Path("/raid/users/popcornell/LibriSpeech"),
         output_dir=Path(lhotse_manifest_dir), dataset_parts=LIBRISPLITs)

    # OPTIONAL: add possibility of using forced alignment to split further the utterances
    # before extracting CUTs

    all_cuts = []
    for split in LIBRISPLITs:
        c_rec = lhotse.load_manifest(Path(lhotse_manifest_dir) / f"librispeech_recordings_{split}.jsonl.gz")
        c_sup = lhotse.load_manifest(Path(lhotse_manifest_dir) / f"librispeech_supervisions_{split}.jsonl.gz")
        # patch recordings to add num_channels
        #c_sup = split_with_alignment(c_sup)
        c_cut = CutSet.from_manifests(recordings=c_rec,
                                      supervisions=c_sup)
        all_cuts.append(c_cut)

    all_cuts = combine_manifests(all_cuts)
    # Generate conversational dataset
    conversation_cuts = create_librispeech_conversation_dataset(
        librispeech_cuts=all_cuts,
         num_conversations=1000,
         num_utterances_per_conversation=30,
         num_speakers_per_conversation=8,
         output_dir=Path("./conversations"))

    # Save the cuts
    conversation_cuts.to_file("conversation_cuts.jsonl.gz")

    print("Multi-speaker conversation simulator ready!")
    print("Use create_librispeech_conversation_dataset() to generate conversations.")