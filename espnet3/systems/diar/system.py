"""Diarization system for ESPnet3.

On top of the generic :class:`BaseSystem` stages (``collect_stats``, ``train``,
``infer``, ``measure``) this system adds:

* ``data_preparation`` -- config-driven simulated-mixture / AMI-cut preparation
  (``training_config.data_prep.func``).
* ``infer_longform`` / ``measure_longform`` -- full-session (long-form)
  diarization with the streaming speaker cache, writing one RTTM per session and
  scoring collar-based DER. These are config-driven via the ``longform`` block of
  the inference / metrics configs and are the reproducible path to the
  session-level numbers.
"""

import json
import logging
import time
from importlib import import_module
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class DiarizationSystem(BaseSystem):
    """Sortformer diarization system.

    No tokenizer is needed. Provides a ``data_preparation`` stage that delegates
    to a recipe-provided function (see ``src/data_prep.py``).
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            stage_log_mapping={
                "data_preparation": "training_config.data_dir",
                "infer_longform": "inference_config.inference_dir",
                "measure_longform": "metrics_config.inference_dir",
            },
            **kwargs,
        )

    def data_preparation(self, *args, **kwargs):
        """Run the recipe's data-preparation callable, if configured.

        Expects ``training_config.data_prep`` with a ``func`` dotted path, e.g.::

            data_prep:
              func: src.data_prep.prepare
              output_dir: ${data_dir}/synth
              ...
        """
        self._reject_stage_args("data_preparation", args, kwargs)
        cfg = getattr(self.training_config, "data_prep", None)
        if cfg is None or not getattr(cfg, "func", None):
            logger.info(
                "DiarizationSystem.data_preparation(): no `data_prep.func` "
                "configured; skipping (assuming manifests already exist)."
            )
            return
        start = time.perf_counter()
        module_path, func_name = cfg.func.rsplit(".", 1)
        func = getattr(import_module(module_path), func_name)
        func_kwargs = {k: v for k, v in cfg.items() if k != "func"}
        logger.info("Running data preparation via %s", cfg.func)
        func(**func_kwargs)
        logger.info("Data preparation completed in %.2fs", time.perf_counter() - start)

    # ------------------------------------------------------------------ #
    # long-form (full-session) diarization
    # ------------------------------------------------------------------ #
    def _longform_out_dir(self, cfg):
        out = getattr(cfg, "out_dir", None)
        if out is None:
            base = getattr(self.inference_config, "inference_dir", None) or "."
            out = str(Path(base) / "longform")
        return out

    def infer_longform(self, *args, **kwargs):
        """Full-session diarization → one hyp/ref RTTM per session.

        Config (``inference_config.longform``)::

            longform:
              mode: streaming            # streaming (speaker cache) | stitch
              ami_dir: /path/to/AMI
              cond: sdm                  # sdm | mdm | ihm-mix
              splits: [dev]
              nemo_ckpt: /path/v2.pt     # or ckpt / hf_model / nest
              threshold: 0.5
              chunk_sec: 90
              overlap_sec: 30
        """
        self._reject_stage_args("infer_longform", args, kwargs)
        import lhotse

        from espnet2.diar.sortformer.longform import (
            build_eval_model,
            run_longform_inference,
        )

        cfg = getattr(self.inference_config, "longform", None)
        assert cfg is not None, "inference_config.longform must be set"
        import torch

        device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
        model = build_eval_model(
            nemo_ckpt=getattr(cfg, "nemo_ckpt", None),
            ckpt=getattr(cfg, "ckpt", None),
            hf_model=getattr(cfg, "hf_model", "nvidia/diar_sortformer_4spk-v1"),
            nest=getattr(cfg, "nest", None),
            num_spk=getattr(cfg, "num_spk", 4),
            device=device,
        )
        out_dir = self._longform_out_dir(cfg)
        ami_dir = Path(cfg.ami_dir)
        cond = getattr(cfg, "cond", "sdm")
        splits = (
            OmegaConf.to_container(cfg.splits) if hasattr(cfg, "splits") else ["dev"]
        )
        for split in splits:
            rec = lhotse.load_manifest(
                str(ami_dir / f"ami-{cond}_recordings_{split}.jsonl.gz")
            )
            sup = lhotse.load_manifest(
                str(ami_dir / f"ami-{cond}_supervisions_{split}.jsonl.gz")
            )
            cuts = lhotse.CutSet.from_manifests(recordings=rec, supervisions=sup)
            split_out = str(Path(out_dir) / f"{cond}_{split}")
            logger.info(
                "Long-form inference: cond=%s split=%s mode=%s -> %s",
                cond,
                split,
                getattr(cfg, "mode", "streaming"),
                split_out,
            )
            run_longform_inference(
                model,
                cuts,
                split_out,
                mode=getattr(cfg, "mode", "streaming"),
                device=device,
                chunk_sec=getattr(cfg, "chunk_sec", 90.0),
                overlap_sec=getattr(cfg, "overlap_sec", 30.0),
                threshold=getattr(cfg, "threshold", 0.5),
                log=logger.info,
            )

    def measure_longform(self, *args, **kwargs):
        """Score collar-based session-level DER over the long-form RTTMs.

        Config (``metrics_config.longform``): ``cond``, ``splits``, ``collar``
        (default 0.25), optional ``out_dir`` (defaults to the inference
        ``longform`` directory).
        """
        self._reject_stage_args("measure_longform", args, kwargs)
        from espnet2.diar.sortformer.longform import score_rttm_der

        cfg = getattr(self.metrics_config, "longform", None)
        assert cfg is not None, "metrics_config.longform must be set"
        out_dir = self._longform_out_dir(cfg)
        cond = getattr(cfg, "cond", "sdm")
        splits = (
            OmegaConf.to_container(cfg.splits) if hasattr(cfg, "splits") else ["dev"]
        )
        collar = getattr(cfg, "collar", 0.25)
        results = {}
        for split in splits:
            split_out = str(Path(out_dir) / f"{cond}_{split}")
            res = score_rttm_der(split_out, collar=collar)
            results[f"{cond}_{split}"] = res
            logger.info("Long-form DER [%s %s]: %s", cond, split, res)
        out_json = Path(out_dir) / "longform_der.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s", out_json)
        return results
