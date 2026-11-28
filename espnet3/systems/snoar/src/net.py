"""The separation network SNOAR trains: an ESPnet2 enhancement model.

``EnhNet`` wraps an ``espnet2.enh.espnet_model.ESPnetEnhancementModel`` so the
training code sees one call, ``net(wav) -> (B, J, T)``. Keeping the ESPnet2
model intact means the weights SNOAR trains are an ordinary ESPnet2
enhancement checkpoint: ``save_net`` writes the ``config.yaml`` and
``model.pth`` that ``EnhancementTask.build_model_from_file`` and
``espnet2.bin.enh_inference.SeparateSpeech`` read.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml

from espnet3.systems.snoar.src.ops import apply_mixture_consistency

logger = logging.getLogger(__name__)

ENH_TASK = "espnet2.tasks.enh.EnhancementTask"
CONFIG_NAME = "config.yaml"
MODEL_NAME = "model.pth"


# STFT-encoder separators that run under bfloat16 autocast.
BF16_STFT_SEPARATORS = ("bsrnn",)


class EnhNet(nn.Module):
    """ESPnet2 enhancement model returning every output slot as one tensor.

    Args:
        enh_model: The ESPnet2 enhancement model.
        enh_config: The ESPnet2 training config that builds ``enh_model``.
            It is written next to the weights by :func:`save_net`.
        mix_consistency: ``none``, ``uniform``, ``global`` or ``energy``. Any
            mode but ``none`` projects the outputs so they sum to the input
            (Wisdom et al. 2019); the MixIT recipe of the reference uses
            ``energy``.
        fs: Sampling rate of the audio, passed to the encoder. STFT encoders
            scale their window by ``fs / default_fs``; a model trained with
            ``default_fs`` other than ``fs`` (e.g. the URGENT 2025 TF-GridNet,
            ``default_fs: 8000``) runs at the wrong resolution without it.
            ``None`` passes nothing.
        fp32: Run the network with autocast disabled. ``None`` decides from
            the config: models with ``encoder: stft`` whose separator does
            complex arithmetic in its own layers (no bfloat16 kernels) run in
            float32. BSRNN (its complex mask is applied in float32 inside the
            layer), TF-GridNet with its built-in STFT (``encoder: same``) and
            waveform/conv-encoder models follow the trainer's precision.

    Examples:
        >>> net = build_net({"separator": "tfgridnetv2",
        ...                  "separator_conf": {"n_srcs": 2, "n_fft": 64,
        ...                  "stride": 32, "n_layers": 1, "emb_dim": 4,
        ...                  "lstm_hidden_units": 8, "attn_n_head": 1}})
        >>> net(torch.randn(1, 4000)).shape
        torch.Size([1, 2, 4000])
    """

    def __init__(
        self,
        enh_model: nn.Module,
        enh_config: Dict[str, Any],
        mix_consistency: str = "none",
        fs: Optional[int] = None,
        fp32: Optional[bool] = None,
    ) -> None:
        """Wrap ``enh_model``."""
        super().__init__()
        if mix_consistency not in ("none", "uniform", "global", "energy"):
            raise ValueError(
                f"mix_consistency={mix_consistency!r}: expected none, uniform, "
                "global or energy"
            )
        self.enh = enh_model
        self.enh_config = enh_config
        self.mix_consistency = mix_consistency
        self.fs = fs
        if fp32 is None:
            fp32 = (
                enh_config.get("encoder", "same") == "stft"
                and enh_config.get("separator") not in BF16_STFT_SEPARATORS
            )
        self.fp32 = fp32

    @property
    def num_outputs(self) -> int:
        """Return the number of output slots of the network."""
        return int(self.enh.num_spk)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Separate ``wav`` (B, T) into ``(B, J, T)``."""
        if wav.dim() != 2:
            raise ValueError(f"EnhNet expects (batch, samples), got {tuple(wav.shape)}")
        lengths = torch.full(
            (wav.shape[0],), wav.shape[1], dtype=torch.long, device=wav.device
        )
        kwargs = {} if self.fs is None else {"fs": self.fs}
        if self.fp32:
            with torch.autocast(device_type=wav.device.type, enabled=False):
                outs, *_ = self.enh.forward_enhance(wav.float(), lengths, **kwargs)
        else:
            outs, *_ = self.enh.forward_enhance(wav, lengths, **kwargs)
        out = torch.stack([o[..., : wav.shape[1]] for o in outs], dim=1)
        if self.mix_consistency != "none":
            out = apply_mixture_consistency(out, wav, mode=self.mix_consistency)
        return out


def _complete_enh_config(enh_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill the ESPnet2 defaults that a waveform-domain model does not need."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(enh_config):
        enh_config = OmegaConf.to_container(enh_config, resolve=True)
    conf = copy.deepcopy(dict(enh_config))
    conf.setdefault("encoder", "same")
    conf.setdefault("decoder", "same")
    if conf["encoder"] == "same":
        conf.setdefault("encoder_conf", {})
    if conf["decoder"] == "same":
        conf.setdefault("decoder_conf", {})
    return conf


def build_net(
    enh_config: Dict[str, Any],
    mix_consistency: str = "none",
    fs: Optional[int] = None,
) -> EnhNet:
    """Build a freshly initialized network from an ESPnet2 enhancement config.

    Args:
        enh_config: ``encoder``/``separator``/``decoder`` (+ ``*_conf``)
            exactly as an ESPnet2 ``enh.sh`` training config spells them.
            ``encoder`` and ``decoder`` default to ``same``.
        mix_consistency: See :class:`EnhNet`.
        fs: See :class:`EnhNet`.

    Returns:
        The wrapped network.

    Raises:
        ValueError: If ``enh_config`` names no separator.
    """
    import sys
    from argparse import Namespace

    from espnet2.tasks.enh import EnhancementTask

    if not enh_config or "separator" not in enh_config:
        raise ValueError(
            "net config must name an ESPnet2 separator, e.g. "
            "{separator: tfgridnetv2, separator_conf: {n_srcs: 2, ...}}"
        )
    # The full default-merged config, as `enh.sh` writes it, so that
    # `build_model_from_file` can rebuild the model from `config.yaml`.
    argv, sys.argv = sys.argv, ["snoar"]
    try:
        full = EnhancementTask.get_default_config()
    finally:
        sys.argv = argv
    full.update(_complete_enh_config(enh_config))
    enh = EnhancementTask.build_model(Namespace(**full))
    return EnhNet(enh, full, mix_consistency=mix_consistency, fs=fs)


def resolve_model_dir(source: str) -> str:
    """Return a local exported-model directory for ``source`` when there is one.

    A local directory is returned as is. A Hugging Face repository holding an
    exported SNOAR model (a directory with ``snoar.yaml``, ``config.yaml`` and
    ``model.pth``, as in an ESPnet3 ``pack_model`` bundle) is downloaded, and
    the local path of that directory is returned. Anything else, such as an
    ESPnet2 enhancement model tag, is returned unchanged for
    ``espnet_model_zoo`` to resolve.

    Args:
        source: Local directory or Hugging Face repository id.

    Returns:
        A local directory, or ``source`` itself.

    Raises:
        ValueError: If the repository holds more than one exported model.

    Examples:
        >>> resolve_model_dir("/no/such/dir_and_not_a_repo")
        '/no/such/dir_and_not_a_repo'
    """
    if Path(source).is_dir() or source.count("/") != 1 or source.startswith("/"):
        return source
    try:
        from huggingface_hub import list_repo_files, snapshot_download

        files = list_repo_files(source)
    except Exception:  # not a reachable HF repo: leave it to espnet_model_zoo
        return source
    dirs = sorted(
        {
            str(Path(f).parent)
            for f in files
            if Path(f).name == "snoar.yaml"
            and str(Path(f).parent / MODEL_NAME) in files
        }
    )
    if not dirs:
        return source
    if len(dirs) > 1:
        raise ValueError(f"{source} holds several exported models: {dirs}")
    pattern = "*" if dirs[0] == "." else f"{dirs[0]}/*"
    root = snapshot_download(source, allow_patterns=[pattern])
    return str(Path(root) / dirs[0])


def _resolve_pretrained(source: str) -> tuple[Path, Path]:
    """Return ``(config, model_file)`` for a local directory or an HF tag."""
    path = Path(source)
    if path.is_dir():
        config, model = path / CONFIG_NAME, path / MODEL_NAME
        if not config.is_file() or not model.is_file():
            raise FileNotFoundError(
                f"{path} is not an exported model: expected {CONFIG_NAME} and "
                f"{MODEL_NAME} (run the `export` stage or `espnet snoar adapt`)"
            )
        return config, model
    try:
        from espnet_model_zoo.downloader import ModelDownloader
    except ImportError as e:  # pragma: no cover - a core espnet dependency
        raise RuntimeError("espnet_model_zoo is needed to download models") from e
    files = ModelDownloader().download_and_unpack(source)
    return Path(files["train_config"]), Path(files["model_file"])


def load_net(
    source: str,
    mix_consistency: Optional[str] = None,
    fs: Optional[int] = None,
) -> EnhNet:
    """Load a trained ESPnet2 enhancement model as an :class:`EnhNet`.

    Args:
        source: A directory written by :func:`save_net` or a Hugging Face
            model tag (e.g. ``kohei0209/tfgridnet_urgent25``).
        mix_consistency: Override the mode stored in the directory's
            ``snoar.yaml``; ``None`` keeps the stored mode (``none`` for an
            ESPnet2 model that has no such file).
        fs: See :class:`EnhNet`.

    Returns:
        The wrapped network, in train mode on CPU.
    """
    from espnet2.tasks.enh import EnhancementTask

    config, model_file = _resolve_pretrained(source)
    enh, _ = EnhancementTask.build_model_from_file(config, model_file, "cpu")
    with open(config, encoding="utf-8") as f:
        enh_config = yaml.safe_load(f)
    sidecar = Path(config).parent / "snoar.yaml"
    stored = {}
    if sidecar.is_file():
        with open(sidecar, encoding="utf-8") as f:
            stored = yaml.safe_load(f) or {}
    mode = mix_consistency or stored.get("mix_consistency", "none")
    net = EnhNet(enh, enh_config, mix_consistency=mode, fs=fs)
    net.train()
    logger.info(
        "Loaded %s (%d outputs, mix_consistency=%s)", source, net.num_outputs, mode
    )
    return net


def save_net(net: EnhNet, out_dir: str | Path, extra: Dict[str, Any]) -> Path:
    """Write ``net`` as an ESPnet2 enhancement model directory.

    Args:
        net: The network to write.
        out_dir: Output directory; created if missing.
        extra: Written to ``snoar.yaml``; must hold ``readout`` so the
            directory can be enhanced with the right output read.

    Returns:
        The output directory.

    Raises:
        ValueError: If ``extra`` has no ``readout``.
    """
    if "readout" not in extra:
        raise ValueError("snoar.yaml needs a `readout` (dnf, mixit or raw)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.enh.state_dict(), out / MODEL_NAME)
    with open(out / CONFIG_NAME, "w", encoding="utf-8") as f:
        yaml.safe_dump(net.enh_config, f, sort_keys=False)
    meta = {"mix_consistency": net.mix_consistency, **extra}
    with open(out / "snoar.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return out
