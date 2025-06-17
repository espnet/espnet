from pathlib import Path
from typing import Dict, Iterable, List, Union, Optional, Tuple
import torch
import numpy as np
import librosa
import tempfile
import soundfile as sf
import os
from transformers import AutoProcessor
from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer

class Qwen2AudioTokenizer(AbsTokenizer):
    """Qwen2-Audio tokenizer that handles both text and audio inputs"""
    
    @typechecked
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    ):
        self.model_name = model_name
        
        # Initialize processor lazily to avoid pickling issues
        self.processor = None
        
    def _build_processor(self):
        """Build AutoProcessor lazily to avoid serialization issues"""
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def text2tokens(self, line: str) -> List[str]:
        """Convert text to tokens using Qwen2-Audio processor"""
        self._build_processor()
        
        # For text-only input, use standard tokenization
        tokens = self.processor.tokenizer.tokenize(line)
        return tokens
    
    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Convert tokens back to text"""
        self._build_processor()
        return self.processor.tokenizer.convert_tokens_to_string(list(tokens))
    
    def create_multimodal_query(
        self,
        text_input: str,
        audio_input: Optional[Tuple[List[np.ndarray], int]] = None,
    ) -> Dict:
        """
        Create query with both text and audio inputs following Qwen2-Audio format
        This is the core tokenization process from your example
        """
        self._build_processor()
        
        def create_tempfile(audio_input: Tuple[List[np.ndarray], int]) -> List[str]:
            n_audios, sr = len(audio_input[0]), audio_input[1]
            tempfiles = [tempfile.NamedTemporaryFile(suffix='.wav', delete=False) 
                        for _ in range(n_audios)]
            for temp, wav in zip(tempfiles, audio_input[0]):
                sf.write(temp.name, wav, sr)
            return [tempfile.name for tempfile in tempfiles]

        def delete_tempfile(p_files: List[str]):
            for p_file in p_files:
                os.unlink(p_file)

        # Handle audio input if provided
        if audio_input is not None:
            temp_files = create_tempfile(audio_input)
            
            # Prepare multimodal query
            wavs_query = [
                {'type': 'audio', 'audio_url': file} 
                for file in temp_files if os.path.exists(file)
            ]
            
            text_query = [{'type': 'text', 'text': text_input}]
            
            query = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': wavs_query + text_query}
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                query, add_generation_prompt=True, tokenize=False
            )
            
            # Load audio files
            audios = [
                librosa.load(
                    file, 
                    sr=self.processor.feature_extractor.sampling_rate
                )[0] 
                for file in temp_files if os.path.exists(file)
            ]
            
            # Process inputs with both text and audio
            inputs = self.processor(
                text=text, 
                audios=audios, 
                sampling_rate=self.processor.feature_extractor.sampling_rate, 
                return_tensors="np", 
                padding=True
            )
            delete_tempfile(temp_files)
        else:
            # Text-only processing
            inputs = self.processor(
                text=text_input, 
                return_tensors="np", 
                padding=True
            )
        
        return inputs
