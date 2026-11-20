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

import logging
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

logger = logging.getLogger(__name__)

FREEZE_CHOICES = ("none", "encoder", "decoder")

#: ``discriminator_params`` keys of the multi-scale discriminator branch that
#: espnet2's ``DACDiscriminator`` had until 9209eb47036 (Aug 2024). The
#: published ESPnet DAC codecs (``espnet/libritts_dac_24k``,
#: ``espnet/libritts_dac_16k``, espnet 202402) still carry them in their
#: ``config.yaml`` although their checkpoints never held weights for that
#: branch, so the current class rejects their config.
LEGACY_DAC_DISCRIMINATOR_KEYS = (
    "scales",
    "scale_downsample_pooling",
    "scale_downsample_pooling_params",
    "scale_discriminator_params",
)

COMPRESSION_MODEL_MAP = {
    "cosine_similarity": CosineSimilarityCompression,
    "density_peak": DensityPeakCompression,
    "dp_segmentation": DPSegmentationCompression,
    # No-op: every frame becomes its own segment.  Use this as the
    # "no compression" reference when evaluating codec reconstruction.
    "none": IdentityCompression,
    "identity": IdentityCompression,
}


def _convert_to_container(config: Any) -> Any:
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

    Example:
        .. code-block:: python

            >>> model = build_compression_model("dp_segmentation", max_span=4)
            >>> type(model).__name__
            'DPSegmentationCompression'
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

    Example:
        .. code-block:: python

            >>> artifacts = read_bundle_artifacts("exp/train_encodec/model_pack")
            >>> sorted(artifacts)
            ['inference_config', 'metrics_config', 'model_file',
             'publication_config', 'training_config']
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

    Example:
        .. code-block:: python

            >>> from espnet_model_zoo.downloader import ModelDownloader

            >>> downloader = ModelDownloader()
            >>> tag = "espnet/libritts_encodec_24k"
            >>> artifacts = downloader.download_and_unpack(tag)
            >>> codec = build_codec_from_bundle(artifacts)
            >>> type(codec.codec.generator.quantizer).__name__
            'ResidualVectorQuantizer'
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
    model_config = drop_legacy_dac_discriminator_keys(model_config)
    model = get_espnet_model(task, model_config)
    state_dict = read_model_state(artifacts["model_file"])
    model.load_state_dict(fill_missing_snake_alpha(model, state_dict), strict=True)
    return model


def drop_legacy_dac_discriminator_keys(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a DAC ``model_config`` without the removed multi-scale keys.

    A no-op for any other codec, and for a DAC config written against the
    current ``DACDiscriminator``. For a config of the published 2024 DAC
    models it drops :data:`LEGACY_DAC_DISCRIMINATOR_KEYS` from
    ``codec_conf.discriminator_params`` with a warning; the checkpoint holds
    no weights for that branch, so the discriminator built from the
    remaining keys is the one it was saved from.

    Args:
        model_config: A ``GANCodecTask`` model config (``codec`` plus
            ``codec_conf``), as a plain dict.

    Returns:
        ``model_config`` itself when nothing applies, else a copy with the
        keys removed.

    Example:
        .. code-block:: python

            >>> config = {"codec": "dac", "codec_conf": {"discriminator_params": {
            ...     "scales": 3, "msmpmb_discriminator_params": {}}}}
            >>> drop_legacy_dac_discriminator_keys(config)["codec_conf"]
            {'discriminator_params': {'msmpmb_discriminator_params': {}}}
    """
    if model_config.get("codec") != "dac":
        return model_config
    params = (model_config.get("codec_conf") or {}).get("discriminator_params") or {}
    legacy = [key for key in LEGACY_DAC_DISCRIMINATOR_KEYS if key in params]
    if not legacy:
        return model_config
    logger.warning(
        "Dropping discriminator_params %s of a DAC config written before "
        "espnet2 removed the multi-scale discriminator branch (the published "
        "espnet/libritts_dac_* models); the checkpoint has no weights for it.",
        legacy,
    )
    config = dict(model_config)
    config["codec_conf"] = dict(model_config["codec_conf"])
    config["codec_conf"]["discriminator_params"] = {
        key: value for key, value in params.items() if key not in legacy
    }
    return config


def fill_missing_snake_alpha(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Add the Snake ``alpha`` entries an older checkpoint does not carry.

    espnet2's ``Snake1d.alpha`` was a fixed tensor of ones, never saved,
    until 094c5f188ef (Oct 2025) made it a parameter. A checkpoint from
    before (the published DAC models) therefore lacks every ``*.alpha``
    entry the current model has; they are filled with ones, the value the
    checkpoint was trained with, so the load can stay strict and the model
    is the one that was published. A saved ``alpha`` is left alone, and
    nothing else is added.

    Args:
        model: The model the checkpoint is loaded into.
        state_dict: The checkpoint's model-level state dict.

    Returns:
        ``state_dict`` itself when nothing is missing, else a copy with the
        missing ``alpha`` entries added.

    Example:
        .. code-block:: python

            >>> state_dict = read_model_state("81epoch.pth")
            >>> state_dict = fill_missing_snake_alpha(model, state_dict)
            >>> model.load_state_dict(state_dict, strict=True)
    """
    missing = [
        key
        for key, value in model.state_dict().items()
        if key.endswith(".alpha") and key not in state_dict and value.numel() == 1
    ]
    if not missing:
        return state_dict
    logger.warning(
        "Filling %d Snake `alpha` entries missing from the checkpoint with 1.0 "
        "(saved before espnet2 made alpha a parameter).",
        len(missing),
    )
    filled = dict(state_dict)
    for key in missing:
        filled[key] = torch.ones_like(model.state_dict()[key])
    return filled


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

    The checkpoint is read with :func:`read_model_state` (Lightning
    checkpoints and the legacy ``model.`` prefix are unwrapped). Unlike
    espnet2's own ``strict=False`` loading, which silently yields a random
    model when keys do not match, the load is always strict, so any
    mismatch raises.

    Args:
        model: Module that receives the weights, modified in place.
        model_file: Path to a model-level state dict (``.pth``) or a
            Lightning checkpoint (``.ckpt``).

    Raises:
        RuntimeError: If the checkpoint keys do not match ``model`` exactly.

    Example:
        .. code-block:: python

            >>> load_model_state_strict(model, "exp/train_encodec_libritts/last.ckpt")
    """
    model.load_state_dict(read_model_state(model_file), strict=True)


def read_model_state(model_file: str) -> Dict[str, torch.Tensor]:
    """Read a checkpoint as a model-level state dict.

    Lightning checkpoints (``last.ckpt``) nest the weights under a
    ``state_dict`` key, and older ``ESPnetLightningModule`` versions
    prefixed the inner model with ``model.``; both are unwrapped here.

    Args:
        model_file: Path to a model-level state dict (``.pth``) or a
            Lightning checkpoint (``.ckpt``).

    Returns:
        The model-level state dict, on the CPU.

    Example:
        .. code-block:: python

            >>> sorted(read_model_state("exp/train_encodec_libritts/last.ckpt"))[:1]
            ['codec.discriminator.discriminators.0.convs.0.conv.bias']
    """
    state_dict = torch.load(model_file, map_location="cpu")
    if "state_dict" in state_dict:  # Lightning checkpoint layout
        state_dict = state_dict["state_dict"]
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    return state_dict


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

    Example:
        .. code-block:: python

            >>> freeze_codec_module(model, "encoder")
            >>> any(p.requires_grad for p in model.codec.generator.encoder.parameters())
            False
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

    Example:
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

        .. code-block:: python

            >>> model = LACE(
            ...     compression_model={
            ...         "name": "cosine_similarity",
            ...         "eval_rate": 0.5,
            ...     },
            ...     pretrained_model_tag="espnet/libritts_encodec_24k",
            ... )
            >>> type(model.codec.generator.quantizer).__name__
            'CompressionResidualVectorQuantizer'
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
        """Build the base codec, load its weights, then wrap its quantizer.

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
                ``NewGame/libritts_codec_train_encodec_libritts``) or an
                espnet2 zoo model (e.g. ``espnet/libritts_encodec_24k``).
            pretrained_model_dir: Local bundle directory with a ``meta.yaml``:
                an unpacked zoo model or an espnet3 ``pack_model`` output.
            codec: Codec name for a fresh, untrained ``GANCodecTask`` model
                (e.g. ``encodec``). Mutually exclusive with the
                ``pretrained_*`` sources.
            codec_conf: ``codec_conf`` block used together with ``codec``.
            task: espnet2 task path used to build the base model for
                ``codec`` and for espnet2 zoo bundles (whose ``config.yaml``
                does not name a task).
            freeze_codec_module_name: ``none``, ``encoder`` or ``decoder``.

        Raises:
            ValueError: If not exactly one model source
                (``pretrained_model_tag``, ``pretrained_model_dir``,
                ``codec``) is given, if the bundle layout is unknown, or if
                the compression model name or freeze choice is unknown.
            TypeError: If the codec's quantizer is not the shared
                ``ResidualVectorQuantizer`` (e.g. HiFiCodec's group
                quantizer).
            RuntimeError: If the bundle's ``model_file`` does not match the
                base model's keys exactly.
        """
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
