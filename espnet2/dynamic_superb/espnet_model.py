from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml

# from .modeling_qwen2 import Qwen2AudioForConditionalGeneration
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from typeguard import typechecked

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.beam_search import BeamSearch

from .qwen2_scorer import Qwen2HFScorer


class ESPnetQwen2AudioModel(AbsESPnetModel):
    """ESPnet model integrating Qwen2-Audio from transformers"""

    @typechecked
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        vocab_size: int = 50000,
        token_list: Union[Tuple[str, ...], List[str]] = (),
        ignore_id: int = -1,
        decode_config_path: Optional[str] = None,
        use_espnet_beam_search: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.ignore_id = ignore_id
        self.token_list = token_list
        self.use_espnet_beam_search = use_espnet_beam_search

        # Load Qwen2-Audio model and processor using standard transformers approach
        self.qwen2audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, device_map="cpu", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Get actual vocabulary size from the model
        self.vocab_size = self.qwen2audio_model.config.vocab_size
        print(f"Using vocabulary size: {self.vocab_size}")

        # Decode configuration
        self.decode_config = {
            "beam_size": 1,
            "penalty": 0.0,
            "maxlenratio": 0.0,
            "minlenratio": 0.0,
            "ctc_weight": 0.0,
            "lm_weight": 0.0,
            "normalize_length": False,
        }

        if decode_config_path is not None:
            try:
                with open(decode_config_path, "r") as f:
                    decode_config = yaml.safe_load(f)
                self.decode_config.update(decode_config)
                print(f"Loaded decode config: {decode_config}")
            except FileNotFoundError:
                print(
                    f"Warning: decode config file {decode_config_path} not found, using defaults"
                )

        # If beam search is enabled, ensure beam_size > 1
        if self.use_espnet_beam_search and self.decode_config["beam_size"] <= 1:
            print(
                "Warning: beam_size <= 1 with ESPnet beam search enabled, setting beam_size=5"
            )
            self.decode_config["beam_size"] = 5

        # Initialize ESPnet beam search if enabled
        if self.use_espnet_beam_search and self.decode_config["beam_size"] > 1:
            print(
                f"Initializing ESPnet beam search with beam_size={self.decode_config['beam_size']}, vocab_size={self.vocab_size}"
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
        self.scorer.to(speech.device)
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

        if self.use_espnet_beam_search:
            # Check if beam search is properly initialized
            return self._inference_with_espnet_beam(
                input_ids, attention_mask, input_features, feature_attention_mask
            )
        else:
            return self._inference_with_hf_generate(
                input_ids, attention_mask, input_features, feature_attention_mask
            )

    def _inference_with_espnet_beam(
        self, input_ids, attention_mask, input_features, feature_attention_mask
    ):
        scorer = Qwen2HFScorer(
            self.qwen2audio_model,
            input_ids,
            attention_mask,
            input_features,
            feature_attention_mask,
        )

        beam_search = BeamSearch(
            beam_size=self.decode_config["beam_size"],
            vocab_size=self.qwen2audio_model.config.vocab_size,
            sos=self.qwen2audio_model.language_model.config.bos_token_id,
            eos=self.qwen2audio_model.language_model.config.eos_token_id,
            scorers={"decoder": scorer},
            weights={"decoder": 1.0},
            normalize_length=False,
        )

        # Run beam search
        dummy_input = torch.zeros(
            input_ids.shape[1], dtype=torch.float, device=input_ids.device
        )
        nbest = beam_search(
            dummy_input,
            maxlenratio=self.decode_config["maxlenratio"],
            minlenratio=self.decode_config["minlenratio"],
        )  # input "xs" not used
        best = nbest[0]

        out_ids = best.yseq

        prediction = self.processor.decode(
            out_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return prediction

    def _inference_with_hf_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> str:
        """Inference using Hugging Face generate method"""

        # Generate response
        if self.decode_config["maxlenratio"] > 0.0:
            input_length = input_ids.size(-1)
            max_length = int(self.decode_config["maxlenratio"] * input_length)
        else:
            max_length = 256
        if self.decode_config["minlenratio"] > 0.0:
            min_length = int(self.decode_config["minlenratio"] * input_length)
        else:
            min_length = 0

        with torch.no_grad():
            pred_ids = self.qwen2audio_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                do_sample=False,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                num_beams=self.decode_config["beam_size"],
                length_penalty=self.decode_config["penalty"],
            )

        # Extract only generated tokens
        pred_ids = pred_ids[:, input_ids.size(1) :]

        # Decode prediction
        prediction = self.processor.batch_decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return prediction
