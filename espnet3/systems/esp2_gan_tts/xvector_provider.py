"""Provider for x-vector (speaker embedding) extraction."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)

# espnet2's ``tts.sh`` defaults (``spk_embed_tool`` / ``spk_embed_model``).
DEFAULT_TOOLKIT = "espnet"
DEFAULT_PRETRAINED_MODEL = "espnet/voxcelebs12_rawnet3"


class XVectorProvider(EnvironmentProvider):
    """Provider for speaker embedding extraction.

    This provider builds a speaker embedding model and audio reader
    for use with XVectorRunner to extract x-vectors in parallel.

    Supported toolkits (``training_config.xvector.toolkit``):
        - espnet (default): espnet2's ``Speech2Embedding``. Needs no extra
          dependency; the default model is ``espnet/voxcelebs12_rawnet3``,
          the same default as espnet2's ``tts.sh``.
        - speechbrain: SpeechBrain's pre-trained models (``pip install
          speechbrain``).
        - rawnet: RawNet3 from the vendored ``RawNet3`` module.

    Examples:
        Paired with
        :class:`~espnet3.systems.esp2_gan_tts.xvector_runner.XVectorRunner`
        by the ``compute_xvectors`` stage:
        ```python
        provider = XVectorProvider(
            training_config,
            params={
                "manifest_path": "data/manifest/train.tsv",
                "output_dir": "data/x_vectors/train",
            },
        )
        env = provider.build_env_local()
        XVectorRunner.forward(0, **env)
        ```
    """

    def __init__(self, config: DictConfig, params: Dict[str, Any] | None = None):
        """Initialize the provider.

        Args:
            config: Configuration with an ``xvector`` section (toolkit,
                pretrained_model, device).
            params: Per-split parameters forwarded from the driver to
                workers: ``manifest_path`` and ``output_dir``.

        Returns:
            None

        Examples:
            ```python
            XVectorProvider(
                training_config,
                params={
                    "manifest_path": "data/manifest/train.tsv",
                    "output_dir": "data/x_vectors/train",
                },
            )
            ```
        """
        super().__init__(config)
        self.params = params or {}

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver, for local execution.

        Returns:
            A dictionary containing the loaded model and manifest data
            needed by ``XVectorRunner.forward``.

        Raises:
            RuntimeError: If required xvector configuration is missing or
                no utterances are available in params.
            ImportError: If the configured toolkit is not installed.

        Examples:
            ```python
            env = provider.build_env_local()
            sorted(env)
            # -> ['config', 'device', 'model', 'output_dir',
            #     'speaker_to_utterances', 'toolkit', 'utterances']
            ```
        """
        return XVectorProvider._build_env(self.config, self.params)

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create a worker setup function for distributed execution.

        Returns:
            A zero-arg callable executed once per worker that returns the
            environment dictionary consumed by ``XVectorRunner.forward``.

        Notes:
            Unlike :meth:`build_env_local`, the model is loaded inside the
            worker, so the (unpicklable) model never crosses a process
            boundary.

        Examples:
            ```python
            setup = provider.build_worker_setup_fn()
            env = setup()  # runs once per worker
            ```
        """
        config = self.config
        params = self.params

        def setup() -> Dict[str, Any]:
            """Build the worker environment from the captured config/params."""
            return XVectorProvider._build_env(config, params)

        return setup

    @staticmethod
    def _build_env(config: DictConfig, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config/params, load the model and manifest, make output_dir."""
        xvector_config = config.get("xvector", None)
        if xvector_config is None:
            raise RuntimeError(
                "xvector configuration not found in training_config. "
                "Please ensure training_config.xvector is set."
            )

        toolkit = xvector_config.get("toolkit", DEFAULT_TOOLKIT)
        pretrained_model = xvector_config.get(
            "pretrained_model", DEFAULT_PRETRAINED_MODEL
        )
        # An explicit ``device: null`` means "pick automatically", like a
        # missing key.
        device = xvector_config.get("device", None) or (
            "cuda:0" if XVectorProvider._has_cuda() else "cpu"
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

    @staticmethod
    def _load_manifest(manifest_path):
        r"""Parse a TSV manifest into utterances + speaker mapping.

        Each line is expected to be ``utt_id\twav_path\ttext\tspeaker_id``.
        Blank lines are skipped; a row with fewer than four columns raises
        ``RuntimeError`` naming the manifest and the line.
        """
        utterances = []
        speaker_to_utterances: Dict[str, list] = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    raise RuntimeError(
                        f"Malformed manifest line in {manifest_path} "
                        f"(expected utt_id/wav_path/text/speaker_id): {line!r}"
                    )
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
            ImportError: If the toolkit's optional package is not installed.
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
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError as e:
                raise ImportError(
                    "xvector.toolkit is 'speechbrain' but speechbrain is not "
                    "installed. Run `pip install speechbrain`, or set "
                    "xvector.toolkit to 'espnet' (no extra dependency)."
                ) from e

            return EncoderClassifier.from_hparams(
                source=pretrained_model,
                run_opts={"device": device},
            )

        elif toolkit == "rawnet":
            import torch

            try:
                from RawNet3 import RawNet3
                from RawNetBasicBlock import Bottle2neck
            except ImportError as e:
                raise ImportError(
                    "xvector.toolkit is 'rawnet' but the RawNet3 modules are "
                    "not importable. Add the RawNet3 checkout to PYTHONPATH "
                    "(see egs2/TEMPLATE/tts1/tts.sh), or set xvector.toolkit "
                    "to 'espnet' (no extra dependency)."
                ) from e

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
                    weights_only=True,
                )["model"]
            )
            model.to(device).eval()
            return model

        else:
            raise ValueError(f"Unknown toolkit: {toolkit}")
