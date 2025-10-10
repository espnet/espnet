from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml
from typeguard import typechecked

from espnet2.legacy.nets.beam_search import BeamSearch
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from .qwen2_scorer import Qwen2HFScorer

try:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2AudioForConditionalGeneration,
    )
    from transformers.modeling_utils import no_init_weights

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


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
        pytest_mode: Optional[bool] = False,
    ):
        super().__init__()
        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.model_name = model_name
        self.ignore_id = ignore_id
        self.token_list = token_list

        if not pytest_mode:
            # Load Qwen2-Audio model and processor using standard transformers approach
            self.qwen2audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.audio_config.encoder_layers = 2
            config.audio_config.d_model = 256
            config.audio_config.encoder_attention_heads = 4
            config.audio_config.encoder_ffn_dim = 256
            config.audio_config.num_hidden_layers = 4
            config.text_config.hidden_size = 128
            config.text_config.num_hidden_layers = 4
            config.text_config.num_attention_heads = 4
            config.text_config.num_key_value_heads = 4
            config.text_config.intermediate_size = 256
            with no_init_weights():
                self.qwen2audio_model = Qwen2AudioForConditionalGeneration(config)

        self.qwen2audio_model.to("cpu")
        self.qwen2audio_model.eval()
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
                    f"Warning: decode config file {decode_config_path}"
                    " not found, using defaults"
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
        """Forward pass required by AbsESPnetModel interface

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

        # Use force_gatherable for DataParallel compatibility
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
        """Collect features for statistics computation"""
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
