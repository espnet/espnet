"""Diarization system for ESPnet3.

On top of the generic :class:`BaseSystem` stages (``collect_stats``, ``train``,
``infer``, ``measure``) this system adds:

* ``data_preparation`` -- config-driven simulated-mixture / AMI-cut preparation
  (``training_config.data_preparation.func``).
* ``infer_longform`` / ``measure_longform`` -- full-session (long-form)
  diarization with the streaming speaker cache, writing one RTTM per session and
  scoring collar-based DER. These are config-driven via the ``longform`` block of
  the inference / metrics configs and are the reproducible path to the
  session-level numbers.

A recipe's ``run.py`` selects this system with::

    from sortformer.system import DiarizationSystem

Stages are then driven from the command line, e.g.::

    python run.py --stages data_preparation train \\
        --training_config conf/training.yaml
    python run.py --stages infer_longform measure_longform \\
        --training_config conf/training.yaml \\
        --inference_config conf/inference.yaml \\
        --metrics_config conf/metrics.yaml
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

    A :class:`BaseSystem` subclass that orchestrates the diarization recipe. No
    tokenizer is needed. In addition to the inherited stages it provides three
    custom ones:

    * ``data_preparation``: delegates to the dotted-path callable named in
      ``training_config.data_preparation.func`` (e.g. ``src.data_prep.prepare``).
    * ``infer_longform``: full-session diarization with the streaming speaker
      cache, configured by ``inference_config.longform``.
    * ``measure_longform``: collar-based session-level DER over the long-form
      RTTMs, configured by ``metrics_config.longform``.

    Used by a recipe's ``run.py`` as::

        from sortformer.system import DiarizationSystem

    Example invocation of the long-form path::

        python run.py --stages infer_longform measure_longform \\
            --training_config conf/training.yaml \\
            --inference_config conf/inference.yaml \\
            --metrics_config conf/metrics.yaml
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        """Initialize the system and register the custom-stage log mapping.

        Args:
            training_config: Parsed training config (provides ``data_dir`` and
                the optional ``data_preparation`` block).
            inference_config: Parsed inference config (provides ``inference_dir``
                and the ``longform`` block).
            metrics_config: Parsed metrics config (provides the scoring
                ``longform`` block).
            **kwargs: Forwarded to :class:`BaseSystem`.
        """
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

        The ``data_preparation`` stage. It imports the dotted-path callable in
        ``training_config.data_preparation.func`` and calls it with every other key in
        the ``data_preparation`` block forwarded as keyword arguments. If no
        ``data_preparation.func`` is configured the stage is a no-op (manifests are
        assumed to already exist).

        Expects ``training_config.data_preparation`` with a ``func`` dotted path, e.g.::

            data_preparation:
              func: src.data_prep.prepare
              output_dir: ${data_dir}/synth
              ...

        Invoke with::

            python run.py --stages data_preparation \\
                --training_config conf/training.yaml

        Args:
            *args: Not accepted; passing positional stage args raises.
            **kwargs: Not accepted; passing keyword stage args raises.
        """
        self._reject_stage_args("data_preparation", args, kwargs)
        cfg = getattr(self.training_config, "data_preparation", None)
        if cfg is None or not getattr(cfg, "func", None):
            logger.info(
                "DiarizationSystem.data_preparation(): no `data_preparation.func` "
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
        """Resolve the long-form output directory.

        Uses ``cfg.out_dir`` if set; otherwise falls back to a ``longform``
        subdirectory under the inference config's ``inference_dir``.
        """
        out = getattr(cfg, "out_dir", None)
        if out is None:
            base = getattr(self.inference_config, "inference_dir", None) or "."
            out = str(Path(base) / "longform")
        return out

    def infer_longform(self, *args, **kwargs):
        """Full-session diarization → one hyp/ref RTTM per session.

        The ``infer_longform`` stage. For every requested condition/split it
        loads the AMI Lhotse manifests, builds an evaluation model from the
        configured checkpoint (``nemo_ckpt`` / ``ckpt`` / ``hf_model`` /
        ``nest``), runs chunked long-form inference, and writes one hypothesis
        and one reference RTTM per session under the output directory.

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
              # out_dir: optional; defaults to <inference_dir>/longform

        Invoke (typically together with ``measure_longform``) with::

            python run.py --stages infer_longform measure_longform \\
                --training_config conf/training.yaml \\
                --inference_config conf/inference.yaml \\
                --metrics_config conf/metrics.yaml

        Args:
            *args: Not accepted; passing positional stage args raises.
            **kwargs: Not accepted; passing keyword stage args raises.

        Raises:
            AssertionError: If ``inference_config.longform`` is not set.
        """
        self._reject_stage_args("infer_longform", args, kwargs)
        import lhotse

        from .longform import (
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

        The ``measure_longform`` stage. For every requested condition/split it
        scores the hypothesis vs. reference RTTMs written by ``infer_longform``
        with a forgiveness collar, logs the per-split DER, writes the aggregate
        to ``longform_der.json`` in the output directory, and returns it.

        Config (``metrics_config.longform``): ``cond``, ``splits``, ``collar``
        (default 0.25), optional ``out_dir`` (defaults to the inference
        ``longform`` directory)::

            longform:
              cond: sdm
              splits: [dev]
              collar: 0.25

        Invoke (after ``infer_longform``) with::

            python run.py --stages infer_longform measure_longform \\
                --training_config conf/training.yaml \\
                --inference_config conf/inference.yaml \\
                --metrics_config conf/metrics.yaml

        Args:
            *args: Not accepted; passing positional stage args raises.
            **kwargs: Not accepted; passing keyword stage args raises.

        Returns:
            A mapping of ``"<cond>_<split>"`` to its DER result dict.

        Raises:
            AssertionError: If ``metrics_config.longform`` is not set.
        """
        self._reject_stage_args("measure_longform", args, kwargs)
        from .longform import score_rttm_der

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
