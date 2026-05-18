"""Provider for x-vector (speaker embedding) extraction."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)


class XVectorProvider(EnvironmentProvider):
    """Provider for speaker embedding extraction.
    
    This provider builds a speaker embedding model and audio reader
    for use with XVectorRunner to extract x-vectors in parallel.
    
    Supported toolkits:
        - espnet: Uses espnet2's Speech2Embedding model
        - speechbrain: Uses SpeechBrain's pre-trained models
        - rawnet: Uses RawNet3 model
    """

    def __init__(self, config: DictConfig):
        """Initialize the provider.
        
        Args:
            config: Configuration with xvector settings
        """
        super().__init__(config)

    def build_env_local(self) -> Dict[str, Any]:
        """Build environment once on driver for local execution.
        
        This method is called once on the driver process before any workers are spawned.
        It loads the manifest file, extracts utterance and speaker information, and prepares the parameters needed for worker setup.

        Returns:
            A dictionary containing model configuration and manifest data for worker setup.
        
        Raises:
            RuntimeError: If the manifest file is not found or contains no utterances, or if required xvector configuration is missing.
        """
        config = self.config
        xvec_cfg = config.get("xvector", None)
        toolkit = xvec_cfg.get("toolkit", None)  
        pretrained_model = xvec_cfg.get("pretrained_model", None)
        device = xvec_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")

        if xvec_cfg is None:
            raise RuntimeError("xvector configuration not found in training_config. Please ensure training_config.xvector is set with necessary parameters.")
        if toolkit is None or pretrained_model is None:
            raise RuntimeError("training_config.xvector.toolkit and training_config.xvector.pretrained_model must be set for compute_xvectors stage.")
        
        # Load model
        model = self._build_model(toolkit, pretrained_model, device)
        
        # Get utterances and speaker mapping from params (loaded from manifest)
        utterances = self.params.get("utterances", [])
        speaker_to_utterances = self.params.get("speaker_to_utterances", {})

        if not utterances:
            raise RuntimeError("No utterances found in manifest. Please ensure the manifest file is correctly generated and contains valid entries.")
        
        return {
            "model": model,
            "toolkit": toolkit,
            "device": device,
            "utterances": utterances,
            "speaker_to_utterances": speaker_to_utterances,
            "config": self.config,
        }

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create a worker setup function for distributed execution.
        
        This function returns a callable that initializes the speaker embedding model and loads the manifest data for each worker process.
        The setup function is called once per worker process to prepare the environment for x-vector extraction.

        Returns:
            A callable that sets up the worker environment and returns necessary objects for x-vector extraction.
        
        Raises:
            RuntimeError: If required xvector configuration is missing.
        """
        config = self.config
        
        def setup() -> Dict[str, Any]:
            xvec_cfg = config.get("xvector", None)
            toolkit = xvec_cfg.get("toolkit", None)
            pretrained_model = xvec_cfg.get("pretrained_model", None)
            device = xvec_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")
            
            if xvec_cfg is None:
                raise RuntimeError("xvector configuration not found in training_config. Please ensure training_config.xvector is set with necessary parameters.")
            
            if toolkit is None or pretrained_model is None:
                raise RuntimeError("training_config.xvector.toolkit and training_config.xvector.pretrained_model must be set for compute_xvectors stage.")
            
            # Load model
            model = self._build_model(toolkit, pretrained_model, device)
            
            # Get utterances and speaker mapping from params (loaded from manifest)
            utterances = params.get("utterances", [])
            speaker_to_utterances = params.get("speaker_to_utterances", {})

            if not utterances:
                raise RuntimeError("No utterances found in manifest. Please ensure the manifest file is correctly generated and contains valid entries.")
            
            return {
                "model": model,
                "toolkit": toolkit,
                "device": device,
                "utterances": utterances,
                "speaker_to_utterances": speaker_to_utterances,
                "config": config,
            }
        
        return setup

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _build_model(toolkit: str, pretrained_model: str, device: str):
        """Build the speaker embedding model.
        
        This method migrates from extract_spk_embed.py, provided by espnet2.
        Args:
            toolkit: Type of toolkit ('espnet', 'speechbrain', 'rawnet')
            pretrained_model: Path or ID of the pre-trained model
            device: Device to load model on ('cuda:0', 'cpu', etc.)
            
        Returns:
            Loaded model object
        
        Raises:
            ValueError: If an unknown toolkit is specified.
        """
        if toolkit == "espnet":
            from espnet2.bin.spk_inference import Speech2Embedding
            
            # Determine if this is a HuggingFace model or local path
            if pretrained_model.endswith(".pth"):
                model_tag = None
                model_file = pretrained_model
            else:
                model_tag = pretrained_model
                model_file = None
            
            model = Speech2Embedding.from_pretrained(
                model_tag=model_tag,
                model_file=model_file,
                batch_size=1,
                dtype="float32",
                train_config=None,
            )
            return model
            
        elif toolkit == "speechbrain":
            from speechbrain.pretrained import EncoderClassifier
            
            model = EncoderClassifier.from_hparams(
                source=pretrained_model,
                run_opts={"device": device}
            )
            return model
            
        elif toolkit == "rawnet":
            import torch
            from RawNet3 import RawNet3
            from RawNetBasicBlock import Bottle2neck
            
            model = RawNet3(
                Bottle2neck,
                model_scale=8,
                context=True,
                summed=True,
                encoder_type="ECA",
                nOut=256,
                out_bn=False,
                sinc_stride=10,
                log_sinc=True,
                norm_sinc="mean",
                grad_mult=1,
            )
            model.load_state_dict(
                torch.load(
                    pretrained_model,
                    map_location=lambda storage, loc: storage,
                )["model"]
            )
            model.to(device).eval()
            return model
        else:
            raise ValueError(f"Unknown toolkit: {toolkit}")
