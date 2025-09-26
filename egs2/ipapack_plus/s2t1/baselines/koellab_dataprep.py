"""
Kaldi-style dataset processor for ESPNet training.
Usage:
    python dataprep/koellab.py \
        --dataset "KoelLabs/EpaDB" \
        --output_dir /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_epadb \
        --cache_dir /work/nvme/bbjs/sbharadwaj/hf_datasets
    
    python dataprep/koellab.py \
        --dataset "KoelLabs/SpeechOceanNoTH" \
        --output_dir  /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_speechoceannotth \
        --cache_dir /work/nvme/bbjs/sbharadwaj/hf_datasets
"""

import os
import json
import argparse
import librosa
import soundfile as sf
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datasets import Dataset, load_dataset
from tqdm import tqdm
import regex as re


class TaskTokens:
    """Task tokens for different training tasks"""
    ASR = "<asr>"
    PR = "<pr>"
    G2P = "<g2p>"
    P2G = "<p2g>"
    TEXT_NA = "<na>"
    NO_TIME = "<notimestamps>"
    UNK_LANG = "<UNK_LANG>"
    DEFAULT_LANG = "<eng>"


class KaldiFileWriter:
    """Utility class for writing Kaldi-style files"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_scp_file(self, filename: str, items: List[Tuple[str, str]]):
        """Write .scp file (space-separated key-value pairs)"""
        items = sorted(items, key=lambda x: x[0])  # sort by key
        with open(self.output_dir / filename, 'w') as f:
            for key, value in items:
                f.write(f"{key} {value}\n")

    def write_text_file(self, filename: str, items: List[Tuple[str, str]]):
        """Write text file (space-separated key-value pairs)"""
        items = sorted(items, key=lambda x: x[0])  # sort by key
        with open(self.output_dir / filename, 'w') as f:
            for key, value in items:
                f.write(f"{key} {value}\n")
    
    def write_json_file(self, filename: str, data: Dict[str, Any]):
        """Write JSON file"""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors"""
    
    def __init__(self, output_dir: str, sample_rate: int = 16000):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.file_writer = KaldiFileWriter(output_dir)
        
    @abstractmethod
    def load_dataset(self, split: str) -> Dataset:
        pass
    
    @abstractmethod
    def extract_fields(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_language_tag(self, sample: Dict[str, Any]) -> str:
        """Get language tag for sample"""
        pass
    
    def text_normalization(self, text: str) -> str:
        """Normalize text by removing punctuation and symbols"""
        # Remove punctuation and symbols, keep only alphanumeric and spaces
        normalized = re.sub(r'[^\p{L}\p{N}\s]', '', text)
        # Collapse multiple spaces and strip
        return ' '.join(normalized.split())
    
    def format_phonemes(self, phonemes: str) -> str:
        """Format phonemes with forward slashes"""
        return "".join([f"/{char}/" for char in phonemes])

    def save_audio(self, audio_array: Any, sr: int, utt_id: str) -> Tuple[str, int]:
        """Save audio data and return file path
        """
        audio_array = audio_array.detach().cpu().numpy()
        audio_array = audio_array.T  # -> (T, C)
        num_samples = audio_array.shape[0]

        if sr != self.sample_rate:
            audio_array = librosa.resample(audio_array.T, orig_sr=sr, target_sr=self.sample_rate).T
            num_samples = audio_array.shape[0]

        audio_dir = Path(self.output_dir) / "recording"
        audio_dir.mkdir(exist_ok=True)

        audio_path = audio_dir / f"{utt_id}.wav"
        sf.write(audio_path, audio_array, self.sample_rate)
        return str(audio_path.absolute()), num_samples
    
    def generate_task_files(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate all task-specific files"""
        
        wav_scp = []
        spk2utt = defaultdict(list)
        utt2spk = []
        utt2num_samples = []
        
        pr_text = []
        orthographic = []
        pr_readable = []
        
        stats = {"total_samples": 0, "total_duration": 0}
        
        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                data_fields = self.extract_fields(sample)
                utt_id = data_fields['utt_id']
                text = data_fields['text']
                phonemes = data_fields['phonemes']

                lang_tag = self.get_language_tag(sample)
                audio_path, num_samples = self.save_audio(data_fields['audio_data'], data_fields['sample_rate'], utt_id)
                duration = data_fields['duration']
                speaker_id = data_fields['speaker_id']

                normalized_text = self.text_normalization(text)
                formatted_phonemes = self.format_phonemes(phonemes)

                wav_scp.append((utt_id, audio_path))
                spk2utt[speaker_id].append(utt_id)
                utt2spk.append((utt_id, speaker_id))
                utt2num_samples.append((utt_id, str(num_samples)))
                
                # text component

                pr_text.append((utt_id, f"{lang_tag}{TaskTokens.PR}{TaskTokens.NO_TIME} {formatted_phonemes}"))
                orthographic.append((utt_id, normalized_text))
                pr_readable.append((utt_id, phonemes))

                stats["total_samples"] += 1
                stats["total_duration"] += duration
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        ######## WRITE ALL FILES #######
        self.file_writer.write_scp_file("wav.scp", wav_scp)
        spk2utt_formatted = [(spk, " ".join(utts)) for spk, utts in spk2utt.items()]
        self.file_writer.write_text_file("spk2utt", spk2utt_formatted)
        self.file_writer.write_text_file("utt2spk", utt2spk)
        self.file_writer.write_text_file("utt2num_samples", utt2num_samples)
        self.file_writer.write_text_file("text", pr_text)
        self.file_writer.write_text_file("text.good", pr_readable)
        self.file_writer.write_text_file("text.asr", orthographic)
        #################################
        return stats
    
    def process_split(self, split: str='full'):
        """Process a dataset split and generate Kaldi files.
        Args:
            split (str): Dataset split to process ('full' means combine all data)
        """
        print(f"Processing {split} split...")
        
        # Load dataset
        samples = self.load_dataset(split)        
        
        # Generate files
        stats = self.generate_task_files(samples)
        
        # Save statistics
        stats['split'] = split
        stats['duration_hours'] = stats['total_duration'] / 3600
        
        self.file_writer.write_json_file("stats.json", stats)
        print(f"Processed {split}: {stats['total_samples']} samples, audio of {stats['duration_hours']:.2f} hours")
        return stats


class EpaDBProcessor(DatasetProcessor):
    """Processor for EpaDB dataset"""

    def __init__(self, output_dir: str, cache_dir: str = None, sampling_rate: int = 16000):
        super().__init__(output_dir, sampling_rate)
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    def load_dataset(self, split: str) -> Dataset:
        """Load EpaDB dataset split"""
        assert split=='full', "EpaDB only supports 'full' split. Combine all data!"
        dataset = load_dataset("KoelLabs/EpaDB")
        train_ds = dataset['train']
        test_ds = dataset['test']
        return list(test_ds)+list(train_ds)

    def extract_fields(self, sample: Dict[str, Any]) -> Tuple[str, Any, str, str]:
        """Extract fields from EpaDB sample"""
        # {'audio': <datasets.features._torchcodec.AudioDecoder object at 0x7f744b099660>, 'ipa': 'lɛtspleɪsɔkə', 'text': " let's play soccer ", 'speaker_code': 'spkr103'}
        speaker = sample['speaker_code']
        utt_id = f"{speaker}-{hash(sample['text'])}"
        audiosample = sample['audio'].get_all_samples()
        assert audiosample.duration_seconds < 20 , 'Audio longer than 10 seconds! Skip!'

        return {
            "utt_id": utt_id,
            "audio_data": audiosample.data,
            'sample_rate': audiosample.sample_rate,
            'duration': audiosample.duration_seconds,
            "text": sample['text'].strip(),
            "phonemes": sample['ipa'].strip(),
            "speaker_id": speaker,
        }
    
    def get_language_tag(self, sample: Dict[str, Any]) -> str:
        """Get language tag (EpaDB is English with spanish accent)"""
        return "<eng>"


class DoReCoProcessor(DatasetProcessor):
    """Processor for DoReCo South England dataset"""

    def __init__(self, output_dir: str, cache_dir: str = None, sampling_rate: int = 16000):
        super().__init__(output_dir, sampling_rate)
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    def load_dataset(self, split: str) -> Dataset:
        """Load EpaDB dataset split"""
        assert split=='full', "EpaDB only supports 'full' split. Combine all data!"
        dataset = load_dataset("KoelLabs/DoReCo")
        train_ds = dataset['train']
        return train_ds

    def extract_fields(self, sample: Dict[str, Any]) -> Tuple[str, Any, str, str]:
        """Extract fields from EpaDB sample"""
        # {'audio': <datasets.features._torchcodec.AudioDecoder object at 0x7f744b099660>, 'ipa': 'lɛtspleɪsɔkə', 'text': " let's play soccer ", 'speaker_code': 'spkr103'}
        speaker = sample['speaker_code']
        utt_id = f"{speaker}-{hash(sample['text'])}"
        audiosample = sample['audio'].get_all_samples()
        assert audiosample.duration_seconds < 20 , 'Audio longer than 20 seconds! Skip!'

        return {
            "utt_id": utt_id,
            "audio_data": audiosample.data,
            'sample_rate': audiosample.sample_rate,
            'duration': audiosample.duration_seconds,
            "text": sample['text'].strip(),
            "phonemes": sample['ipa'].strip(),
            "speaker_id": speaker,
        }
    
    def get_language_tag(self, sample: Dict[str, Any]) -> str:
        """Get language tag (EpaDB is English with spanish accent)"""
        return "<eng>"


class SpeechOceanNoTHProcessor(DatasetProcessor):
    """Processor for SpeechOceanNoTH dataset"""

    def __init__(self, output_dir: str, cache_dir: str = None, sampling_rate: int = 16000):
        super().__init__(output_dir, sampling_rate)
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    def load_dataset(self, split: str) -> Dataset:
        """Load SpeechOceanNoTH dataset split"""
        assert split=='full', "SpeechOceanNoTH only supports 'full' split. Combine all data!"
        dataset = load_dataset("KoelLabs/SpeechOceanNoTH")
        train_ds = dataset['train']
        test_ds = dataset['test']
        return list(train_ds) + list(test_ds)

    def extract_fields(self, sample: Dict[str, Any]) -> Tuple[str, Any, str, str]:
        """Extract fields from EpaDB sample"""
        # {'audio': <datasets.features._torchcodec.AudioDecoder object at 0x7f744b099660>, 'ipa': 'lɛtspleɪsɔkə', 'text': " let's play soccer ", 'speaker_code': 'spkr103'}
        speaker = sample['speaker_code']
        utt_id = f"{speaker}-{hash(sample['text'])}"
        audiosample = sample['audio'].get_all_samples()
        assert audiosample.duration_seconds < 20 , 'Audio longer than 20 seconds! Skip!'

        return {
            "utt_id": utt_id,
            "audio_data": audiosample.data,
            'sample_rate': audiosample.sample_rate,
            'duration': audiosample.duration_seconds,
            "text": sample['text'].strip(),
            "phonemes": sample['ipa'].strip(),
            "speaker_id": speaker,
        }
    
    def get_language_tag(self, sample: Dict[str, Any]) -> str:
        """Get language tag (EpaDB is English with spanish accent)"""
        return "<eng>"


def main():
    parser = argparse.ArgumentParser(description="Generate Kaldi-style files from HuggingFace datasets")
    parser.add_argument("--dataset", type=str, default="KoelLabs/EpaDB", 
                        choices=["KoelLabs/DoReCo",
                                 "KoelLabs/EpaDB",
                                 "KoelLabs/SpeechOceanNoTH"],
                       help="Dataset to process")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for Kaldi files")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="HuggingFace datasets cache directory")
    parser.add_argument("--splits", nargs="+", default=["full"],
                       help="Dataset splits to process")
    
    args = parser.parse_args()
    
    # Create processor based on dataset
    if args.dataset == "KoelLabs/EpaDB":
        processor = EpaDBProcessor(args.output_dir, args.cache_dir)
    elif args.dataset == "KoelLabs/DoReCo":
        processor = DoReCoProcessor(args.output_dir, args.cache_dir)
    elif args.dataset == "KoelLabs/SpeechOceanNoTH":
        processor = SpeechOceanNoTHProcessor(args.output_dir, args.cache_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Process each split
    all_stats = {}
    for split in args.splits:
        try:
            stats = processor.process_split(split)
            all_stats[split] = stats
        except Exception as e:
            print(f"Error processing {split}: {e}")
    
    # Save overall statistics
    overall_output = Path(args.output_dir)
    overall_output.mkdir(parents=True, exist_ok=True)
    with open(overall_output / "overall_stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    main()