from typing import Dict, List, Optional, Tuple, Union

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from typeguard import typechecked

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.torch_utils.device_funcs import force_gatherable

class ESPnetQwen2AudioModel(AbsESPnetModel):
    """ESPnet model integrating Qwen2-Audio from transformers"""
    
    @typechecked
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        vocab_size: int = 50000,
        token_list: Union[Tuple[str, ...], List[str]] = (),
        ignore_id: int = -1,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list
        
        # Load Qwen2-Audio model and processor using standard transformers approach
        self.qwen2audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            device_map="cpu",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # For inference-only, freeze the model parameters
        for param in self.qwen2audio_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass required by AbsESPnetModel interface[9][22]
        
        Returns:
            loss: Scalar tensor (dummy for inference-only)
            stats: Dictionary of statistics for logging
            weight: Batch size for normalization
        """
        batch_size = speech.shape[0]
        
        # Since this is inference-only, return dummy loss
        # In a training scenario, you would compute actual loss here
        loss = torch.tensor(0.0, device=speech.device, requires_grad=True)
        
        stats = {
            "loss": loss.detach(),
            "batch_size": batch_size,
        }
        
        # Use force_gatherable for DataParallel compatibility[26]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    
    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features for statistics computation[30][32]"""
        return {"feats": speech, "feats_lengths": speech_lengths}
    
    def inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        **kwargs,
    ) -> str:
        """Custom inference method using Qwen2-Audio"""
        # Generate response
        with torch.no_grad():
            pred_ids = self.qwen2audio_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=256,
            )
            
        # Extract only generated tokens
        pred_ids = pred_ids[:, input_ids.size(1):]
        
        # Decode prediction
        prediction = self.processor.batch_decode(
            pred_ids, 
            skip_special_tokens=True
        )[0]
        
        return prediction