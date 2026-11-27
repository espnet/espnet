"""SNOAR system: label-free speech enhancement and self-adaptation.

Stages, in the order ``all`` runs them:

``create_dataset``
    Build the recipe's manifests (shared ``BaseSystem`` stage).
``harvest_noise``
    Write an in-domain noise pool from the speech-free gaps of the training
    recordings (VAD labels or an energy VAD). Needed by DNF and MixIT when no
    external noise corpus is given.
``train``
    Train one of ``supervised``, ``dnf``, ``mixit``, ``snoar``, ``remixit``
    (shared ``BaseSystem`` stage; the objective is the model's ``mode``).
``export``
    Write the trained network (the EMA teacher for SNOAR/RemixIT) as an
    ESPnet2 enhancement model directory plus ``snoar.yaml``, which seeds the
    next training run and is what ``infer`` and ``espnet snoar enhance`` load.
``infer`` / ``measure``
    Enhance the test sets and score them (shared stages).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


def resolve_last_checkpoint(ckpt: Path) -> Path:
    """Return the newest ``last*.ckpt`` when ``ckpt`` is ``last.ckpt``.

    When a run is resumed into a directory that already holds ``last.ckpt``,
    Lightning writes the new last checkpoint as ``last-v1.ckpt`` and leaves
    ``last.ckpt`` at the resume point. Exporting ``last.ckpt`` would then
    silently export the old model.

    Args:
        ckpt: The configured checkpoint.

    Returns:
        ``ckpt``, or the most recently written ``last-v*.ckpt`` beside it.

    Examples:
        >>> resolve_last_checkpoint(Path("exp/x/model.pth"))
        PosixPath('exp/x/model.pth')
    """
    if ckpt.name != "last.ckpt":
        return ckpt
    candidates = [p for p in ckpt.parent.glob("last*.ckpt") if p.exists()]
    if not candidates:
        return ckpt
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    if newest != ckpt:
        logger.warning(
            "export: %s is older than %s; exporting the latter", ckpt, newest
        )
    return newest


class SNOARSystem(BaseSystem):
    """Stages for DNF / MixIT / supervised seeds and SNOAR self-adaptation.

    Examples:
        >>> names = [s.name for s in SNOARSystem.stages]
        >>> names[:6]
        ['create_dataset', 'harvest_noise', 'train', 'export', 'infer', 'measure']
        >>> names[6:]
        ['pack_model', 'upload_model']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="harvest_noise", config="training", log_dir="exp_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="export", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
    )

    def harvest_noise(self, *args, **kwargs):
        """Write the noise pool from the gaps of the training recordings.

        Configure under ``training_config.harvest_noise``:

        .. code-block:: yaml

            harvest_noise:
              manifest: ${data_dir}/manifest/train.jsonl
              vad: energy          # or a .rttm / Kaldi segments file
              out: ${data_dir}/manifest/noise_train.jsonl
              margin: 0.1
              min_gap: 0.3

        Raises:
            RuntimeError: If the section or one of its paths is missing.
        """
        self._reject_stage_args("harvest_noise", args, kwargs)
        from espnet3.systems.snoar.src.data.manifest import (
            read_manifest,
            write_manifest,
        )
        from espnet3.systems.snoar.src.data.vad import harvest_noise

        cfg = self._get_required_config(
            self.training_config,
            "harvest_noise",
            "training_config.harvest_noise must be set for the harvest_noise stage",
        )
        for key in ("manifest", "vad", "out"):
            self._get_required_config(
                cfg, key, f"training_config.harvest_noise.{key} must be set"
            )
        rows = harvest_noise(
            read_manifest(cfg.manifest),
            vad=str(cfg.vad),
            fs=int(cfg.get("fs", 16000)),
            margin=float(cfg.get("margin", 0.1)),
            min_gap=float(cfg.get("min_gap", 0.3)),
        )
        if not rows:
            raise RuntimeError(
                "no noise-only region found; lower harvest_noise.min_gap or give "
                "an external noise corpus as the noise pool instead"
            )
        n = write_manifest(rows, cfg.out)
        logger.info("harvest_noise: wrote %d regions -> %s", n, cfg.out)

    def export(self, *args, **kwargs):
        """Export the trained network as an ESPnet2 enhancement model directory.

        Configure under ``training_config.export``:

        .. code-block:: yaml

            export:
              checkpoint: ${exp_dir}/last.ckpt   # or an averaged *.pth
              out_dir: ${exp_dir}/export

        Raises:
            RuntimeError: If the section is missing.
            FileNotFoundError: If the checkpoint does not exist.
        """
        self._reject_stage_args("export", args, kwargs)
        from espnet3.systems.snoar.src.net import save_net

        cfg = self._get_required_config(
            self.training_config,
            "export",
            "training_config.export must be set for the export stage",
        )
        ckpt = Path(
            cfg.get("checkpoint", Path(self.training_config.exp_dir) / "last.ckpt")
        )
        ckpt = resolve_last_checkpoint(ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt} (run `train` first)")
        model = instantiate(OmegaConf.create(self.training_config.model))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        state = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in state.items()
        }
        model.load_state_dict(state)
        out = save_net(
            model.inference_net(),
            cfg.out_dir,
            extra={
                "readout": model.readout,
                "mode": model.mode,
                "fs": int(self.training_config.get("fs", 16000)),
                "checkpoint": str(ckpt),
                "num_updates": int(model.num_updates),
            },
        )
        logger.info(
            "export: %s (%s, readout=%s, %d updates) -> %s",
            ckpt,
            model.mode,
            model.readout,
            int(model.num_updates),
            out,
        )
        max_steps = self.training_config.get("trainer", {}).get("max_steps")
        if max_steps and int(model.num_updates) < int(max_steps):
            logger.warning(
                "export: %s has %d of trainer.max_steps=%d updates; set "
                "export.checkpoint if this is not the checkpoint you meant",
                ckpt,
                int(model.num_updates),
                int(max_steps),
            )
        return out
