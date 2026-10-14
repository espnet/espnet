"""The LACE model: an espnet2 GAN codec with a layer-adaptive quantizer.

``LACE`` is referenced from a training config as a Hydra ``_target_``. The
config must leave the top-level ``task:`` key unset; ``LACESystem`` then
builds the model with ``instantiate(config.model)``.

The base codec comes from a published model bundle (``pretrained_model_tag``
on the Hugging Face Hub, or ``pretrained_model_dir`` on disk), whose weights
are strict-loaded into the UNWRAPPED base model BEFORE the quantizer is
wrapped, so the checkpoint's natural key layout matches and nothing can
overwrite the weights afterwards. Both bundle layouts that ``espnet_model_zoo``
understands are accepted: an espnet2 zoo model (``meta.yaml`` naming
``train_config`` + ``model_file``, e.g. ``espnet/libritts_encodec_24k``) and an
espnet3 ``pack_model`` output (``training_config`` + ``model_file``).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml
from omegaconf import OmegaConf

from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.systems.lace.src.compression import BaseCompressionModel
from espnet3.systems.lace.src.cosine_similarity import CosineSimilarityCompression
from espnet3.systems.lace.src.density_peak import DensityPeakCompression
from espnet3.systems.lace.src.dp_segmentation import DPSegmentationCompression
from espnet3.systems.lace.src.identity import IdentityCompression
from espnet3.systems.lace.src.quantizer import CompressionResidualVectorQuantizer

FREEZE_CHOICES = ("none", "encoder", "decoder")

COMPRESSION_MODEL_MAP = {
    "cosine_similarity": CosineSimilarityCompression,
    "density_peak": DensityPeakCompression,
    "dp_segmentation": DPSegmentationCompression,
    # No-op: every frame becomes its own segment.  Use this as the
    # "no compression" baseline when evaluating codec reconstruction.
    "none": IdentityCompression,
    "identity": IdentityCompression,
}


def _convert_to_container(config):
    """Convert possible OmegaConf containers from Hydra into plain dicts."""
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def build_compression_model(name: str, **kwargs) -> BaseCompressionModel:
    """Build a compression model from its registered name.

    Args:
        name: Key of ``COMPRESSION_MODEL_MAP``: ``cosine_similarity``,
            ``density_peak``, ``dp_segmentation``, or ``identity`` (alias
            ``none``).
        **kwargs: Constructor arguments of the selected class:

            - ``cosine_similarity``: ``max_tokens_per_group`` (default 4).
            - ``density_peak``: ``k`` (default 5), ``beta`` (default 0.2),
              ``max_span`` (default 4).
            - ``dp_segmentation``: ``max_span`` (default 4).
            - ``identity``: none.

    Returns:
        The constructed compression model.

    Raises:
        ValueError: If ``name`` is not a key of ``COMPRESSION_MODEL_MAP``.

    Examples:
        ```python
        model = build_compression_model("dp_segmentation", max_span=4)
        type(model).__name__  # -> 'DPSegmentationCompression'
        ```
    """
    if name not in COMPRESSION_MODEL_MAP:
        raise ValueError(
            f"Unknown compression model '{name}'. "
            f"Available: {list(COMPRESSION_MODEL_MAP.keys())}"
        )
    return COMPRESSION_MODEL_MAP[name](**kwargs)


def read_bundle_artifacts(bundle_dir: Union[str, Path]) -> Dict[str, str]:
    """Read a model bundle's ``meta.yaml`` into absolute artifact paths.

    Mirrors what ``espnet_model_zoo.downloader.ModelDownloader.download_and_unpack``
    returns for a downloaded model, for a bundle that is already on disk: an
    unpacked espnet2 zoo model, or the ``model_pack`` directory written by an
    espnet3 ``pack_model`` stage.

    Args:
        bundle_dir: Directory containing ``meta.yaml``.

    Returns:
        Dict mapping each key of ``meta.yaml``'s ``files`` and ``yaml_files``
        to its absolute path.

    Raises:
        FileNotFoundError: If ``bundle_dir`` has no ``meta.yaml``.

    Examples:
        ```python
        artifacts = read_bundle_artifacts("exp/train_encodec_libritts/model_pack")
        sorted(artifacts)
        # -> ['inference_config', 'metrics_config', 'model_file',
        #     'publication_config', 'training_config']
        ```
    """
    bundle_dir = Path(bundle_dir)
    meta_path = bundle_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Not a model bundle (no meta.yaml): {bundle_dir}. Expected an unpacked "
            "espnet_model_zoo model or an espnet3 pack_model output directory."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    artifacts = {}
    for section in ("files", "yaml_files"):
        for key, relative_path in (meta.get(section) or {}).items():
            artifacts[key] = str(bundle_dir / relative_path)
    return artifacts


def build_codec_from_bundle(
    artifacts: Dict[str, str], task: str = "espnet2.tasks.gan_codec.GANCodecTask"
) -> AbsGANESPnetModel:
    """Build an unwrapped codec from bundle artifacts and strict-load its weights.

    Args:
        artifacts: Artifact paths as returned by ``read_bundle_artifacts`` or
            ``ModelDownloader.download_and_unpack``. Two layouts are
            understood: an espnet2 zoo model (``train_config``, the
            espnet2-style ``config.yaml``, plus ``model_file``) and an espnet3
            ``pack_model`` bundle (``training_config``, whose ``task`` and
            ``model`` blocks describe the codec, plus ``model_file``).
        task: espnet2 task path used with an espnet2 zoo model, whose
            ``config.yaml`` does not name its task. Ignored for espnet3
            bundles, which carry their own ``task``.

    Returns:
        The codec (an ``AbsGANESPnetModel``) with the bundle's weights.

    Raises:
        ValueError: If ``artifacts`` names neither ``train_config`` nor
            ``training_config``, or has no ``model_file``.
        RuntimeError: If ``model_file`` does not match the built model's keys
            exactly.

    Examples:
        ```python
        from espnet_model_zoo.downloader import ModelDownloader

        artifacts = ModelDownloader().download_and_unpack("espnet/libritts_encodec_24k")
        codec = build_codec_from_bundle(artifacts)
        type(codec.codec.generator.quantizer).__name__
        # -> 'ResidualVectorQuantizer'
        ```
    """
    from espnet3.utils.task_utils import get_espnet_model

    if "model_file" not in artifacts:
        raise ValueError(f"Bundle has no 'model_file' artifact: {sorted(artifacts)}")
    if "train_config" in artifacts:
        # espnet2 zoo model: config.yaml is the task's argparse namespace.
        with open(artifacts["train_config"], encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
    elif "training_config" in artifacts:
        # espnet3 pack_model bundle: the recipe's resolved training config.
        training_config = OmegaConf.load(artifacts["training_config"])
        task = training_config.task
        model_config = OmegaConf.to_container(training_config.model, resolve=True)
        model_config.pop("freeze_param", None)
    else:
        raise ValueError(
            "Bundle names neither 'train_config' (espnet2 zoo model) nor "
            f"'training_config' (espnet3 pack_model bundle): {sorted(artifacts)}"
        )
    model = get_espnet_model(task, model_config)
    load_model_state_strict(model, artifacts["model_file"])
    return model


def _build_base_model(
    codec: Optional[str],
    codec_conf: Optional[Dict[str, Any]],
    task: str,
    pretrained_model_tag: Optional[str],
    pretrained_model_dir: Optional[str],
) -> AbsGANESPnetModel:
    """Build the unwrapped base codec: from a bundle (with weights) or fresh."""
    sources = [
        pretrained_model_tag is not None,
        pretrained_model_dir is not None,
        codec is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Specify exactly one model source: 'pretrained_model_tag' (Hugging "
            "Face model id), 'pretrained_model_dir' (local bundle directory), "
            "or 'codec'+'codec_conf' (fresh, untrained codec)."
        )

    if pretrained_model_tag is not None:
        from espnet_model_zoo.downloader import ModelDownloader

        artifacts = ModelDownloader().download_and_unpack(pretrained_model_tag)
        return build_codec_from_bundle(artifacts, task)

    if pretrained_model_dir is not None:
        return build_codec_from_bundle(
            read_bundle_artifacts(pretrained_model_dir), task
        )

    from espnet3.utils.task_utils import get_espnet_model

    return get_espnet_model(
        task, {"codec": codec, "codec_conf": _convert_to_container(codec_conf) or {}}
    )


def load_model_state_strict(model: torch.nn.Module, model_file: str) -> None:
    """Strictly load a model-level or Lightning checkpoint into ``model``.

    Lightning checkpoints (``last.ckpt``) nest the weights under a
    ``state_dict`` key; they are unwrapped here (and a legacy ``model.``
    key prefix is stripped when present).  Unlike espnet2's own
    ``strict=False`` loading, which silently yields a random model when
    keys do not match, the load is always strict, so any mismatch raises.

    Args:
        model: Module that receives the weights, modified in place.
        model_file: Path to a model-level state dict (``.pth``) or a
            Lightning checkpoint (``.ckpt``).

    Raises:
        RuntimeError: If the checkpoint keys do not match ``model`` exactly.

    Examples:
        ```python
        load_model_state_strict(model, "exp/train_encodec_libritts/last.ckpt")
        ```
    """
    state_dict = torch.load(model_file, map_location="cpu")
    if "state_dict" in state_dict:  # Lightning checkpoint layout
        state_dict = state_dict["state_dict"]
    if any(key.startswith("model.") for key in state_dict):
        # Older ESPnetLightningModule versions prefixed the inner model.
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    model.load_state_dict(state_dict, strict=True)


def freeze_codec_module(model: AbsGANESPnetModel, module: str = "none") -> None:
    """Freeze parts of the codec generator before fine-tuning.

    Only ``none``/``encoder``/``decoder`` are supported: freezing both (or
    ``all``) would leave the recipe's named ``generator`` optimizer without
    any trainable parameters (the EMA codebooks are buffers, not
    parameters), which the optimizer validation rejects.

    Args:
        model: Codec model whose ``codec.generator`` is partly frozen in
            place.
        module: ``none`` (freeze nothing), ``encoder`` or ``decoder``.

    Raises:
        ValueError: If ``module`` is not one of ``FREEZE_CHOICES``.

    Examples:
        ```python
        freeze_codec_module(model, "encoder")
        any(p.requires_grad for p in model.codec.generator.encoder.parameters())
        # -> False
        ```
    """
    if module not in FREEZE_CHOICES:
        raise ValueError(
            f"freeze_codec_module must be one of {FREEZE_CHOICES}, got '{module}'."
        )
    if module == "none":
        return
    for param in getattr(model.codec.generator, module).parameters():
        param.requires_grad = False


class LACE(ESPnetGANCodecModel):
    """A codec whose residual vector quantizer runs at layer-adaptive frame rates.

    The model of a LACE recipe, reached from a training config through
    ``model._target_`` with ``task:`` left unset, as ``F5TTS`` is. It is an
    espnet2 ``ESPnetGANCodecModel`` (so ``LACESystem`` trains it with the
    GAN trainer, and ``encode``/``decode`` are the codec's own) whose base
    codec comes from a published model bundle (``pretrained_model_tag`` or
    ``pretrained_model_dir``) or is built fresh (``codec`` + ``codec_conf``),
    and whose ``codec.generator.quantizer`` is replaced by a
    ``CompressionResidualVectorQuantizer`` around the original one.

    The bundle's weights are strict-loaded into the UNWRAPPED base codec
    before the quantizer is wrapped, so this constructor is the recipe's only
    weight-loading mechanism. Inference rebuilds the same architecture from
    the spec ``LACESystem.train`` writes to ``${exp_dir}/lace_model.yaml``
    (this constructor's keyword arguments) and loads the fine-tuned
    checkpoint on top. The state dict keeps the espnet2 layout
    (``codec.generator...``, ``codec.discriminator...``), with the wrapped
    quantizer under ``codec.generator.quantizer.rvq``.

    Args:
        compression_model: Dict with ``name`` (key of
            ``COMPRESSION_MODEL_MAP``), optional ``params`` (constructor
            arguments of that compression model), and the quantizer
            wrapper's rate settings, all optional: ``min_rate`` (default
            0.2), ``max_rate`` (default 1.0), ``random_rate`` (default
            ``per_quantizer``), ``eval_rate`` (default ``None``).
        pretrained_model_tag: Hugging Face model id of a published codec,
            downloaded through ``espnet_model_zoo``: the repo an espnet3
            ``upload_model`` stage pushed to (e.g.
            ``NewGame/libritts_codec_train_encodec_libritts``) or an espnet2
            zoo model (e.g. ``espnet/libritts_encodec_24k``).
        pretrained_model_dir: Local bundle directory with a ``meta.yaml``:
            an unpacked zoo model or an espnet3 ``pack_model`` output.
        codec: Codec name for a fresh, untrained ``GANCodecTask`` model
            (e.g. ``encodec``). Mutually exclusive with the ``pretrained_*``
            sources.
        codec_conf: ``codec_conf`` block used together with ``codec``.
        task: espnet2 task path used to build the base model for ``codec``
            and for espnet2 zoo bundles (whose ``config.yaml`` does not
            name a task).
        freeze_codec_module_name: ``none``, ``encoder`` or ``decoder``.

    Raises:
        ValueError: If not exactly one model source (``pretrained_model_tag``,
            ``pretrained_model_dir``, ``codec``) is given, if the bundle
            layout is unknown, or if the compression model name or freeze
            choice is unknown.
        TypeError: If the codec's quantizer is not the shared
            ``ResidualVectorQuantizer`` (e.g. HiFiCodec's group quantizer).
        RuntimeError: If the bundle's ``model_file`` does not match the base
            model's keys exactly.

    Examples:
        Wired from ``egs3/libritts/lace/conf/training.yaml`` (no top-level
        ``task:`` key):

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.lace.src.lace.LACE
              pretrained_model_tag: NewGame/libritts_codec_train_encodec_libritts
              freeze_codec_module_name: none
              compression_model:
                name: cosine_similarity
                params: {}
                min_rate: 0.1
                max_rate: 1.0
                random_rate: per_quantizer
                eval_rate: 0.5

        Or directly:
        ```python
        model = LACE(
            compression_model={"name": "cosine_similarity", "eval_rate": 0.5},
            pretrained_model_tag="NewGame/libritts_codec_train_encodec_libritts",
        )
        type(model.codec.generator.quantizer).__name__
        # -> 'CompressionResidualVectorQuantizer'
        ```
    """

    def __init__(
        self,
        compression_model: Dict[str, Any],
        pretrained_model_tag: Optional[str] = None,
        pretrained_model_dir: Optional[str] = None,
        codec: Optional[str] = None,
        codec_conf: Optional[Dict[str, Any]] = None,
        task: str = "espnet2.tasks.gan_codec.GANCodecTask",
        freeze_codec_module_name: str = "none",
    ):
        """Build the base codec, load its weights, then wrap its quantizer."""
        base_model = _build_base_model(
            codec=codec,
            codec_conf=codec_conf,
            task=task,
            pretrained_model_tag=pretrained_model_tag,
            pretrained_model_dir=pretrained_model_dir,
        )
        # Keep the base model's own (already loaded) codec module.
        super().__init__(codec=base_model.codec)

        quantizer = self.codec.generator.quantizer
        if not isinstance(quantizer, ResidualVectorQuantizer):
            raise TypeError(
                "LACE wrapping supports codecs whose generator uses the shared "
                "ResidualVectorQuantizer (SoundStream, Encodec, DAC, FunCodec); "
                f"got {type(quantizer).__name__} (HiFiCodec's group quantizer "
                "would need a dedicated adapter)."
            )

        compression_config = dict(_convert_to_container(compression_model))
        name = compression_config.pop("name")
        params = compression_config.pop("params", None) or {}
        compression = build_compression_model(name, **params)
        self.codec.generator.quantizer = CompressionResidualVectorQuantizer(
            quantizer, compression, **compression_config
        )

        freeze_codec_module(self, freeze_codec_module_name)
