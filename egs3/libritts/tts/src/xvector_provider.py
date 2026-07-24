"""Provider for x-vector (speaker embedding) extraction."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict

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

    def __init__(self, config: DictConfig, params: Dict[str, Any] | None = None):
        """Initialize the provider.

        Args:
            config: Configuration with xvector settings.
            params: Extra parameters (e.g. ``utterances``, ``speaker_to_utterances``)
                forwarded from the driver to workers via the async spec.
        """
        super().__init__(config)
        self.params = params or {}

    def build_env_local(self) -> Dict[str, Any]:
        """Build environment once on driver for local execution.
        Use speechbrain's pre-trained ECAPA-TDNN model by default

        Returns:
            A dictionary containing the loaded model and manifest data
            needed by ``XVectorRunner.forward``.

        Raises:
            RuntimeError: If required xvector configuration is missing or
                no utterances are available in params.
        """
        xvec_cfg = self.config.get("xvector", None)
        if xvec_cfg is None:
            raise RuntimeError(
                "xvector configuration not found in training_config. "
                "Please ensure training_config.xvector is set."
            )

        toolkit = xvec_cfg.get("toolkit", "speechbrain")
        pretrained_model = xvec_cfg.get(
            "pretrained_model", "speechbrain/spkrec-ecapa-voxceleb"
        )

        device = xvec_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")

        model = self._build_model(toolkit, pretrained_model, device)

        manifest_path = self.params.get("manifest_path", None)
        if manifest_path is None:
            raise RuntimeError(
                "Please provide manifest_path obtained from create_dataset stage"
            )
        utterances, speaker_to_utterances = self._load_manifest(manifest_path)
        if not utterances:
            raise RuntimeError(f"No utterances found in manifest: {manifest_path}")

        output_dir = self.params.get("output_dir", None)
        if output_dir is None:
            raise RuntimeError(
                "output_dir must be provided so workers know where "
                "to write per-utterance .pt files."
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "model": model,
            "toolkit": toolkit,
            "device": device,
            "utterances": utterances,
            "speaker_to_utterances": speaker_to_utterances,
            "output_dir": output_dir,
            "config": self.config,
        }

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create a worker setup function for distributed execution.

        Returns:
            A zero-arg callable executed once per worker that returns the
            environment dictionary consumed by ``XVectorRunner.forward``.
        """
        config = self.config
        params = self.params

        def setup() -> Dict[str, Any]:
            xvec_cfg = config.get("xvector", None)
            if xvec_cfg is None:
                raise RuntimeError(
                    "xvector configuration not found in training_config. "
                    "Please ensure training_config.xvector is set."
                )

            toolkit = xvec_cfg.get("toolkit", "speechbrain")
            pretrained_model = xvec_cfg.get(
                "pretrained_model", "speechbrain/spkrec-ecapa-voxceleb"
            )

            device = xvec_cfg.get(
                "device", "cuda:0" if XVectorProvider._has_cuda() else "cpu"
            )

            model = XVectorProvider._build_model(toolkit, pretrained_model, device)

            manifest_path = params.get("manifest_path", None)
            if manifest_path is None:
                raise RuntimeError(
                    "Please provide manifest_path obtained from create_dataset stage"
                )
            utterances, speaker_to_utterances = XVectorProvider._load_manifest(
                manifest_path
            )
            if not utterances:
                raise RuntimeError(f"No utterances found in manifest: {manifest_path}")

            output_dir = params.get("output_dir", None)
            if output_dir is None:
                raise RuntimeError(
                    "output_dir must be provided so workers know where "
                    "to write per-utterance .pt files."
                )
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            return {
                "model": model,
                "toolkit": toolkit,
                "device": device,
                "utterances": utterances,
                "speaker_to_utterances": speaker_to_utterances,
                "output_dir": output_dir,
                "config": config,
            }

        return setup

    @staticmethod
    def _load_manifest(manifest_path):
        """Parse a TSV manifest into utterances + speaker mapping.

        Each line is expected to be ``utt_id\\twav_path\\ttext\\tspeaker_id``.
        Blank lines are skipped.
        """
        utterances = []
        speaker_to_utterances: Dict[str, list] = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                utt_id = parts[0]
                wav_path = parts[1]
                speaker_id = parts[3]
                utterances.append((utt_id, wav_path))
                speaker_to_utterances.setdefault(speaker_id, []).append(utt_id)
        return utterances, speaker_to_utterances

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

        Args:
            toolkit: Type of toolkit ('espnet', 'speechbrain', 'rawnet').
            pretrained_model: Path or ID of the pre-trained model.
            device: Device to load model on ('cuda:0', 'cpu', etc.).

        Returns:
            Loaded model object.

        Raises:
            ValueError: If an unknown toolkit is specified.
        """
        if toolkit == "espnet":
            from espnet2.bin.spk_inference import Speech2Embedding

            if pretrained_model.endswith(".pth"):
                model_tag = None
                model_file = pretrained_model
            else:
                model_tag = pretrained_model
                model_file = None

            return Speech2Embedding.from_pretrained(
                model_tag=model_tag,
                model_file=model_file,
                batch_size=1,
                dtype="float32",
                train_config=None,
            )

        elif toolkit == "speechbrain":
            from speechbrain.inference.classifiers import EncoderClassifier

            return EncoderClassifier.from_hparams(
                source=pretrained_model,
                run_opts={"device": device},
            )

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
