#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis for Extracted Hinglish ASR Data
Analyzes MUCS 2021, LibriSpeech, and SLR103 datasets from extracted directories

Author: Anshul Kumar  
Date: September 21, 2025
"""

import os
import re
import pathlib
import collections
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regular expression for Hindi text detection
HINDI_RE = re.compile(r'[\u0900-\u097F]')

class ExtractedDatasetAnalyzer:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = pathlib.Path(data_dir)
        self.analysis_results = {}
        
    def analyze_all_datasets(self):
        """Analyze all three extracted datasets and generate comprehensive report."""
        logger.info("Starting comprehensive dataset analysis on extracted data...")
        
        # Analyze each dataset
        self.analysis_results['mucs_2021'] = self.analyze_mucs_2021()
        self.analysis_results['librispeech'] = self.analyze_librispeech()
        self.analysis_results['slr103_hindi'] = self.analyze_slr103_hindi()
        
        # Generate final report
        self.generate_final_report()
        
    def analyze_directory_structure(self, directory: pathlib.Path, max_samples: int = 10):
        """Analyze structure of an extracted directory."""
        structure_info = {
            'total_files': 0,
            'audio_files': 0,
            'text_files': 0,
            'directories': [],
            'audio_samples': [],
            'text_samples': []
        }
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return structure_info
        
        # Walk through directory structure
        all_files = []
        for root, dirs, files in os.walk(directory):
            root_path = pathlib.Path(root)
            structure_info['directories'].extend([str(root_path.relative_to(directory) / d) for d in dirs])
            
            for file in files:
                file_path = root_path / file
                all_files.append(file_path)
                structure_info['total_files'] += 1
                
                if file.endswith(('.wav', '.flac', '.mp3')):
                    structure_info['audio_files'] += 1
                elif file.endswith(('.txt', '.trans')) or file == 'text':
                    structure_info['text_files'] += 1
        
        # Sample audio files for analysis
        audio_files = [f for f in all_files if f.suffix in ['.wav', '.flac', '.mp3']]
        for audio_file in audio_files[:max_samples]:
            try:
                info = sf.info(str(audio_file))
                structure_info['audio_samples'].append({
                    'file': str(audio_file.relative_to(directory)),
                    'samplerate': info.samplerate,
                    'channels': info.channels,
                    'duration': info.duration,
                    'format': info.format
                })
            except Exception as e:
                logger.warning(f"Could not analyze audio file {audio_file}: {e}")
        
        # Sample text files for analysis
        text_files = [f for f in all_files if f.suffix in ['.txt', '.trans'] or f.name == 'text']
        for text_file in text_files[:5]:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    structure_info['text_samples'].append({
                        'file': str(text_file.relative_to(directory)),
                        'line_count': len(lines),
                        'sample_lines': lines[:3],
                        'encoding': 'utf-8'
                    })
            except Exception as e:
                logger.warning(f"Could not analyze text file {text_file}: {e}")
        
        return structure_info
    
    def analyze_mucs_2021(self):
        """Analyze extracted MUCS 2021 Hindi-English code-switching dataset."""
        logger.info("Analyzing MUCS 2021 dataset...")
        
        analysis = {
            'dataset_name': 'MUCS 2021 Hindi-English Code-Switching',
            'purpose': 'Primary code-switching data for fine-tuning',
            'train_analysis': {},
            'test_analysis': {},
            'language_distribution': {},
            'critical_validations': {}
        }
        
        mucs_dir = self.data_dir / "mucs2021"
        
        # Analyze training set
        train_dir = mucs_dir / "train"
        if train_dir.exists():
            analysis['train_analysis'] = self.analyze_directory_structure(train_dir)
            analysis['language_distribution'] = self.analyze_language_distribution(train_dir)
        else:
            logger.error(f"MUCS training directory not found: {train_dir}")
        
        # Analyze test set
        test_dir = mucs_dir / "test"
        if test_dir.exists():
            analysis['test_analysis'] = self.analyze_directory_structure(test_dir)
        else:
            logger.error(f"MUCS test directory not found: {test_dir}")
        
        return analysis
    
    def analyze_language_distribution(self, mucs_dir: pathlib.Path):
        """Analyze Hindi vs English word distribution in MUCS dataset."""
        hindi_words = 0
        english_words = 0
        total_utterances = 0
        
        # Look for text file in transcripts subdirectory
        text_file = mucs_dir / "transcripts" / "text"
        
        if not text_file.exists():
            logger.error(f"MUCS text file not found: {text_file}")
            return {
                'total_utterances': 0,
                'total_words': 0,
                'hindi_words': 0,
                'english_words': 0,
                'hindi_percentage': 0,
                'english_percentage': 0
            }
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_utterances += 1
                    # Split by whitespace and analyze each word
                    parts = line.split()
                    if len(parts) > 1:  # Skip utterance ID
                        words = parts[1:]  # First part is utterance ID
                        for word in words:
                            if HINDI_RE.search(word):
                                hindi_words += 1
                            else:
                                english_words += 1
        except Exception as e:
            logger.error(f"Error analyzing language distribution: {e}")
        
        total_words = hindi_words + english_words
        distribution = {
            'total_utterances': total_utterances,
            'total_words': total_words,
            'hindi_words': hindi_words,
            'english_words': english_words,
            'hindi_percentage': (hindi_words / total_words * 100) if total_words > 0 else 0,
            'english_percentage': (english_words / total_words * 100) if total_words > 0 else 0
        }
        
        return distribution
    
    def analyze_librispeech(self):
        """Analyze extracted LibriSpeech train-clean-100 dataset."""
        logger.info("Analyzing LibriSpeech dataset...")
        
        analysis = {
            'dataset_name': 'LibriSpeech train-clean-100',
            'purpose': 'English pre-training data (use 36h subset)',
            'structure_analysis': {},
            'text_case_analysis': {},
            'critical_validations': {}
        }
        
        # LibriSpeech has nested structure
        libri_dir = self.data_dir / "librispeech" / "LibriSpeech" / "train-clean-100"
        
        if libri_dir.exists():
            analysis['structure_analysis'] = self.analyze_directory_structure(libri_dir)
            analysis['text_case_analysis'] = self.analyze_librispeech_text_case(libri_dir)
        else:
            logger.error(f"LibriSpeech directory not found: {libri_dir}")
        
        return analysis
    
    def analyze_librispeech_text_case(self, libri_dir: pathlib.Path):
        """Analyze text case in LibriSpeech (should be uppercase)."""
        uppercase_lines = 0
        lowercase_lines = 0
        mixed_lines = 0
        total_lines = 0
        
        # Find transcript files
        transcript_files = list(libri_dir.rglob("*.trans.txt"))
        
        for transcript_file in transcript_files[:5]:  # Sample first 5 files
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        total_lines += 1
                        # Extract transcript (after utterance ID)
                        parts = line.split(' ', 1)
                        if len(parts) > 1:
                            transcript = parts[1]
                            if transcript.isupper():
                                uppercase_lines += 1
                            elif transcript.islower():
                                lowercase_lines += 1
                            else:
                                mixed_lines += 1
            except Exception as e:
                logger.warning(f"Error analyzing text case in {transcript_file}: {e}")
        
        return {
            'total_lines': total_lines,
            'uppercase_lines': uppercase_lines,
            'lowercase_lines': lowercase_lines,
            'mixed_lines': mixed_lines,
            'is_all_uppercase': uppercase_lines == total_lines and total_lines > 0
        }
    
    def analyze_slr103_hindi(self):
        """Analyze extracted SLR103 Hindi dataset."""
        logger.info("Analyzing SLR103 Hindi dataset...")
        
        analysis = {
            'dataset_name': 'SLR103 Hindi (MUCS Multilingual)',
            'purpose': 'Hindi pre-training data (95h train, requires 8kHz→16kHz upsampling)',
            'train_analysis': {},
            'test_analysis': {},
            'sampling_rate_validation': {},
            'speaker_analysis': {}
        }
        
        slr103_dir = self.data_dir / "slr103_hindi"
        
        # Analyze training set
        train_dir = slr103_dir / "train"
        if train_dir.exists():
            analysis['train_analysis'] = self.analyze_directory_structure(train_dir)
            analysis['sampling_rate_validation'] = self.validate_slr103_sampling_rates(train_dir)
            analysis['speaker_analysis'] = self.analyze_slr103_speakers(train_dir)
        else:
            logger.error(f"SLR103 training directory not found: {train_dir}")
        
        # Analyze test set
        test_dir = slr103_dir / "test"
        if test_dir.exists():
            analysis['test_analysis'] = self.analyze_directory_structure(test_dir)
        else:
            logger.error(f"SLR103 test directory not found: {test_dir}")
        
        return analysis
    
    def validate_slr103_sampling_rates(self, slr103_dir: pathlib.Path):
        """Validate that SLR103 audio is 8kHz as expected."""
        sampling_rates = collections.Counter()
        total_audio_files = 0
        
        # Find audio files
        audio_files = list(slr103_dir.rglob("*.wav"))
        
        for audio_file in audio_files[:20]:  # Sample first 20 audio files
            try:
                info = sf.info(str(audio_file))
                sampling_rates[info.samplerate] += 1
                total_audio_files += 1
            except Exception as e:
                logger.warning(f"Could not analyze audio file {audio_file}: {e}")
        
        return {
            'total_sampled': total_audio_files,
            'sampling_rate_distribution': dict(sampling_rates),
            'is_8khz_confirmed': 8000 in sampling_rates and len(sampling_rates) == 1
        }
    
    def analyze_slr103_speakers(self, slr103_dir: pathlib.Path):
        """Analyze speaker information in SLR103 dataset."""
        speaker_info = {
            'total_speakers': 0,
            'utterances_per_speaker': {},
            'sample_speakers': []
        }
        
        # Look for transcription file
        transcription_file = slr103_dir / "transcription.txt"
        
        if transcription_file.exists():
            try:
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    speaker_counts = collections.Counter()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Extract speaker ID from utterance ID (usually first part)
                        parts = line.split()
                        if parts:
                            utterance_id = parts[0]
                            # Extract speaker ID (first part before underscore)
                            speaker_id = utterance_id.split('_')[0]
                            speaker_counts[speaker_id] += 1
                    
                    speaker_info['total_speakers'] = len(speaker_counts)
                    speaker_info['utterances_per_speaker'] = dict(speaker_counts.most_common(10))
                    speaker_info['sample_speakers'] = list(speaker_counts.keys())[:5]
                    
            except Exception as e:
                logger.error(f"Error analyzing speakers: {e}")
        
        return speaker_info
    
    def generate_final_report(self):
        """Generate comprehensive markdown report."""
        report_path = pathlib.Path("dataset_analysis_report_extracted.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Dataset Analysis Report (Extracted Data)\n")
            f.write("## Hinglish ASR Project - ESPnet2 Preprocessing\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("| Dataset | Purpose | Status | Key Findings |\n")
            f.write("|---------|---------|--------|-------------|\n")
            
            # MUCS 2021 Summary
            mucs = self.analysis_results.get('mucs_2021', {})
            lang_dist = mucs.get('language_distribution', {})
            hindi_pct = lang_dist.get('hindi_percentage', 0)
            english_pct = lang_dist.get('english_percentage', 0)
            
            f.write(f"| MUCS 2021 | CS Fine-tuning | ✅ | ")
            f.write(f"Hindi: {hindi_pct:.1f}%, English: {english_pct:.1f}% |\n")
            
            # LibriSpeech Summary
            libri = self.analysis_results.get('librispeech', {})
            case_analysis = libri.get('text_case_analysis', {})
            is_uppercase = case_analysis.get('is_all_uppercase', False)
            
            f.write(f"| LibriSpeech | English Pre-training | ✅ | ")
            f.write(f"Text case: {'All uppercase' if is_uppercase else 'Mixed case'} |\n")
            
            # SLR103 Summary  
            slr103 = self.analysis_results.get('slr103_hindi', {})
            sr_validation = slr103.get('sampling_rate_validation', {})
            is_8khz = sr_validation.get('is_8khz_confirmed', False)
            speaker_analysis = slr103.get('speaker_analysis', {})
            total_speakers = speaker_analysis.get('total_speakers', 0)
            
            f.write(f"| SLR103 Hindi | Hindi Pre-training | ✅ | ")
            f.write(f"8kHz confirmed, {total_speakers} speakers |\n\n")
            
            # Detailed Analysis for each dataset
            for dataset_key, dataset_data in self.analysis_results.items():
                f.write(f"## {dataset_data.get('dataset_name', dataset_key)}\n\n")
                f.write(f"**Purpose:** {dataset_data.get('purpose', 'N/A')}\n\n")
                
                # Write detailed structure info
                if 'train_analysis' in dataset_data:
                    self.write_structure_analysis(f, dataset_data['train_analysis'], "Training Set")
                
                if 'test_analysis' in dataset_data:
                    self.write_structure_analysis(f, dataset_data['test_analysis'], "Test Set")
                
                # Write specific analysis results
                self.write_specific_analysis(f, dataset_data, dataset_key)
                f.write("\n---\n\n")
            
            # Critical Validations Section
            f.write("## Critical Validations\n\n")
            f.write("### Language Distribution (MUCS 2021)\n")
            if lang_dist:
                f.write(f"- **Hindi words:** {lang_dist.get('hindi_words', 0):,} ({hindi_pct:.2f}%)\n")
                f.write(f"- **English words:** {lang_dist.get('english_words', 0):,} ({english_pct:.2f}%)\n")
                f.write(f"- **Total utterances:** {lang_dist.get('total_utterances', 0):,}\n")
                
                # Validation against expected 73.85% / 26.15%
                expected_hindi = 73.85
                expected_english = 26.15
                hindi_diff = abs(hindi_pct - expected_hindi)
                english_diff = abs(english_pct - expected_english)
                
                validation_status = "✅ VALIDATED" if hindi_diff < 5.0 else "⚠️ DEVIATION"
                f.write(f"- **Validation:** Expected ~73.85% Hindi, found {hindi_pct:.2f}% {validation_status}\n\n")
            
            f.write("### Sampling Rate Validation (SLR103)\n")
            if sr_validation:
                f.write(f"- **8kHz confirmation:** {'✅ Confirmed' if is_8khz else '❌ Mixed rates found'}\n")
                f.write(f"- **Distribution:** {sr_validation.get('sampling_rate_distribution', {})}\n\n")
            
            f.write("### Text Case Validation (LibriSpeech)\n")
            if case_analysis:
                f.write(f"- **All uppercase:** {'✅ Confirmed' if is_uppercase else '❌ Mixed case found'}\n")
                f.write(f"- **Total lines analyzed:** {case_analysis.get('total_lines', 0)}\n\n")
            
            # Data Processing Recommendations
            f.write("## Data Processing Recommendations\n\n")
            f.write("### MUCS 2021 Processing\n")
            f.write("- Extract utterance IDs and transcriptions from `train/transcripts/text`\n")
            f.write("- Match audio files with transcriptions using utterance IDs\n")
            f.write("- Handle Hindi-English mixed text in BPE vocabulary\n\n")
            
            f.write("### LibriSpeech Processing\n")
            f.write("- Convert uppercase transcriptions to lowercase\n")
            f.write("- Extract 36-hour subset for balanced pre-training\n")
            f.write("- Use existing speaker/chapter organization\n\n")
            
            f.write("### SLR103 Hindi Processing\n")
            f.write("- Implement 8kHz→16kHz upsampling for all audio files\n")
            f.write(f"- Preserve speaker information ({total_speakers} speakers identified)\n")
            f.write("- Match audio files in `audio/` directory with `transcription.txt`\n\n")
        
        logger.info(f"Comprehensive analysis report saved to: {report_path}")
    
    def write_structure_analysis(self, f, analysis, section_name):
        """Write structure analysis section to report file."""
        f.write(f"### {section_name} Structure\n")
        f.write(f"- **Total files:** {analysis.get('total_files', 0):,}\n")
        f.write(f"- **Audio files:** {analysis.get('audio_files', 0):,}\n")
        f.write(f"- **Text files:** {analysis.get('text_files', 0):,}\n\n")
        
        # Directory structure
        directories = analysis.get('directories', [])
        if directories:
            f.write("**Directory Structure:**\n")
            for directory in directories[:10]:  # Show first 10 directories
                f.write(f"- `{directory}`\n")
            if len(directories) > 10:
                f.write(f"- ... and {len(directories) - 10} more directories\n")
            f.write("\n")
        
        # Audio samples
        audio_samples = analysis.get('audio_samples', [])
        if audio_samples:
            f.write("**Audio Sample Analysis:**\n")
            for sample in audio_samples[:3]:
                f.write(f"- `{sample['file']}`: {sample['samplerate']}Hz, ")
                f.write(f"{sample['channels']}ch, {sample['duration']:.2f}s\n")
            f.write("\n")
        
        # Text samples
        text_samples = analysis.get('text_samples', [])
        if text_samples:
            f.write("**Text Sample Analysis:**\n")
            for sample in text_samples[:2]:
                f.write(f"- `{sample['file']}`: {sample['line_count']} lines\n")
                for line in sample['sample_lines']:
                    f.write(f"  - `{line[:100]}...`\n")
            f.write("\n")
    
    def write_specific_analysis(self, f, dataset_data, dataset_key):
        """Write dataset-specific analysis results."""
        if dataset_key == 'mucs_2021' and 'language_distribution' in dataset_data:
            lang_dist = dataset_data['language_distribution']
            f.write("### Language Distribution Analysis\n")
            f.write(f"- **Total words analyzed:** {lang_dist.get('total_words', 0):,}\n")
            f.write(f"- **Hindi percentage:** {lang_dist.get('hindi_percentage', 0):.2f}%\n")
            f.write(f"- **English percentage:** {lang_dist.get('english_percentage', 0):.2f}%\n\n")
        
        elif dataset_key == 'librispeech' and 'text_case_analysis' in dataset_data:
            case_analysis = dataset_data['text_case_analysis']
            f.write("### Text Case Analysis\n")
            f.write(f"- **Uppercase lines:** {case_analysis.get('uppercase_lines', 0)}\n")
            f.write(f"- **Lowercase lines:** {case_analysis.get('lowercase_lines', 0)}\n")
            f.write(f"- **Mixed case lines:** {case_analysis.get('mixed_lines', 0)}\n\n")
        
        elif dataset_key == 'slr103_hindi':
            if 'sampling_rate_validation' in dataset_data:
                sr_validation = dataset_data['sampling_rate_validation']
                f.write("### Sampling Rate Validation\n")
                f.write(f"- **Files sampled:** {sr_validation.get('total_sampled', 0)}\n")
                f.write(f"- **Rate distribution:** {sr_validation.get('sampling_rate_distribution', {})}\n\n")
            
            if 'speaker_analysis' in dataset_data:
                speaker_analysis = dataset_data['speaker_analysis']
                f.write("### Speaker Analysis\n")
                f.write(f"- **Total speakers:** {speaker_analysis.get('total_speakers', 0)}\n")
                utterances_per_speaker = speaker_analysis.get('utterances_per_speaker', {})
                if utterances_per_speaker:
                    f.write("- **Top speakers by utterance count:**\n")
                    for speaker, count in list(utterances_per_speaker.items())[:5]:
                        f.write(f"  - {speaker}: {count} utterances\n")
                f.write("\n")

def main():
    """Main execution function."""
    analyzer = ExtractedDatasetAnalyzer()
    analyzer.analyze_all_datasets()
    print("\n✅ Dataset analysis complete! Check 'dataset_analysis_report_extracted.md' for full results.")

if __name__ == "__main__":
    main()