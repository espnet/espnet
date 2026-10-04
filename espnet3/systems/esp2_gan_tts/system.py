"""ESPnet2 GAN-TTS system implementation.

This system wraps the espnet2 GAN-TTS task family (VITS, JETS, ...) for
ESPnet3. On top of :class:`~espnet3.systems.tts.system.TTSSystem` it adds
the ``compute_xvectors`` stage, and it routes ``collect_stats`` / ``train``
through a GAN-specific Lightning trainer whenever the configured model is an
``AbsGANESPnetModel``.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.training import _instantiate_model
from espnet3.systems.esp2_gan_tts.gan_trainer import build_gan_trainer
from espnet3.systems.esp2_gan_tts.xvector_provider import XVectorProvider
from espnet3.systems.esp2_gan_tts.xvector_runner import XVectorRunner
from espnet3.systems.tts.system import TTSSystem
from espnet3.utils.task_utils import save_espnet_config

logger = logging.getLogger(__name__)


def _build_trainer(config: DictConfig) -> ESPnet3LightningTrainer:
    """Build the Lightning trainer for a GAN-TTS training config.

    Shadows ``espnet3.systems.base.training._build_trainer`` to add GAN-TTS
    dispatch: GAN-TTS models need a second optimizer and a
    generator/discriminator step schedule, which the plain
    ``ESPnetLightningModule``/``ESPnet3LightningTrainer`` pair cannot express.
    The non-GAN branch is deliberately a copy of the base builder's body rather
    than a delegation to it, so that the model is instantiated exactly once -
    instantiating and discarding a model would advance the global RNG and make
    training depend on whether this dispatch happened.

    Args:
        config (DictConfig): The training config. ``config.model`` (with
            ``config.task``, if set) selects the model; ``config.trainer``,
            ``config.exp_dir`` and ``config.best_model_criterion`` configure
            the trainer.

    Returns:
        ESPnet3LightningTrainer: A ``GANTTSLightningTrainer`` if the model is
        an ``AbsGANESPnetModel``, otherwise a plain ``ESPnet3LightningTrainer``.
    """
    model = _instantiate_model(config)
    if isinstance(model, AbsGANESPnetModel):
        return build_gan_trainer(config, model)

    lit_model = ESPnetLightningModule(model, config)
    return ESPnet3LightningTrainer(
        model=lit_model,
        exp_dir=config.exp_dir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )


class GANTTSSystem(TTSSystem):
    """System for espnet2 GAN-based TTS models (VITS, JETS, ...).

    Inherits the ``remove_long_short`` and ``create_token_list`` stages from
    :class:`~espnet3.systems.tts.system.TTSSystem` and adds:

      - ``compute_xvectors``: per-utterance speaker embeddings for
        multi-speaker conditioning.
      - GAN-aware ``collect_stats`` / ``train``: both build the trainer
        through :func:`_build_trainer`, so an ``AbsGANESPnetModel`` gets the
        two-optimizer ``GANTTSLightningTrainer``.

    Additional stage log paths:
        | Stage                 | Path reference                  |
        |---                   |---                              |
        | compute_xvectors     | training_config.xvector.save_path |

    Examples:
        Wired through ``egs3/TEMPLATE/esp2_gan_tts/run.py``:
        ```python
        from egs3.TEMPLATE.esp2_gan_tts.run import DEFAULT_STAGES, build_parser, main
        from espnet3.systems.esp2_gan_tts.system import GANTTSSystem

        parser = build_parser(stages=DEFAULT_STAGES)
        args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
        main(args=args, system_cls=GANTTSSystem, stages=DEFAULT_STAGES)
        ```
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        """Initialize the system and register the ``compute_xvectors`` log dir.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            **kwargs: Forwarded to :class:`TTSSystem`.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            **kwargs,
        )
        resolved = self._resolve_stage_log_ref("training_config.xvector.save_path")
        if resolved:
            self.stage_log_dirs["compute_xvectors"] = Path(resolved)

    def compute_xvectors(self, *args, **kwargs):
        r"""Compute x-vectors for multiple data splits using parallel execution.

        X-vectors (speaker embeddings) are extracted using a pre-trained
        model for train, valid, and test splits. They can be used as
        speaker conditioning in TTS models.

        This method uses espnet3 manifest files generated by the dataset
        builder. Manifest format: ``utt_id\twav_path\ttext\tspeaker_id``
        (TSV).

        Args:
            *args: Must be empty. Passing any positional argument raises
                ``TypeError`` via ``_reject_stage_args``.
            **kwargs: Must be empty. Passing any keyword argument raises
                ``TypeError`` via ``_reject_stage_args``.

        Returns:
            None.

        Raises:
            TypeError: If any positional or keyword arguments are passed.
            RuntimeError: If required configuration is missing or
                manifest files are not found.

        Notes:
            Configuration should include (under ``training_config.xvector``):
                pretrained_model: Model tag or path. Defaults to
                    ``espnet/voxcelebs12_rawnet3``, the espnet2 default.
                toolkit: ``espnet`` (default), ``speechbrain``, or ``rawnet``.
                save_path: Output directory.
                splits: Splits to process (train, valid, test).
                manifest_paths: Optional split-to-manifest mapping
                    (default: ``data/manifest/{split}.tsv``).
                spk_embed_tag: Name of the per-split output directory
                    (``<save_path>/<spk_embed_tag>_<split>``).
                batch_size: Batch size for processing.
                device: Device to use (default: ``cuda:0`` if available).

        Examples:
            ```bash
            python run.py --stages compute_xvectors \
                --training_config conf/training.yaml
            ```
            writes one embedding per utterance:
            ```text
            data/x_vectors/spkrec-ecapa-voxceleb_train/19_198_000000_000000.pt
            ```
        """
        self._reject_stage_args("compute_xvectors", args, kwargs)
        logger.info("GANTTSSystem.compute_xvectors(): starting x-vector computation")

        # Parse the parallel config early so it applies to the x-vector runner.
        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        xvector_config = self._get_required_config(
            self.training_config,
            "xvector",
            "training_config.xvector must be set for compute_xvectors stage.",
        )
        save_path = Path(
            self._get_required_config(
                xvector_config,
                "save_path",
                "training_config.xvector.save_path must be set for "
                "compute_xvectors stage.",
            )
        )
        save_path.mkdir(parents=True, exist_ok=True)

        # Get list of splits to process (Default: all splits)
        splits = xvector_config.get("splits", ["train", "valid", "test"])

        if isinstance(splits, str):
            splits = [splits]

        manifest_paths = xvector_config.get("manifest_paths", {}) or {}

        logger.info(f"Will process splits: {splits}")
        logger.info(f"Manifest paths: {manifest_paths}")

        # Process each split
        for split in splits:
            logger.info(f"Processing split: {split}")

            manifest_path = manifest_paths.get(split, None)
            if manifest_path is None:
                manifest_path = f"data/manifest/{split}.tsv"
            manifest_path = Path(manifest_path).resolve()
            if not manifest_path.exists():
                raise RuntimeError(
                    f"Manifest file not found for split '{split}': "
                    f"{manifest_path}. Please generate the manifest file "
                    "using the create_dataset stage and ensure the path "
                    "is correct."
                )

            utterances, _ = XVectorProvider._load_manifest(manifest_path)
            n_utts = len(utterances)
            if n_utts == 0:
                raise RuntimeError(f"No utterances found in manifest: {manifest_path}.")

            logger.info(f"Split '{split}': {n_utts} utterances in {manifest_path}")

            batch_size = xvector_config.get("batch_size", None)
            async_mode = xvector_config.get("async_mode", False)
            spk_embed_tag = xvector_config.get("spk_embed_tag", "spk_embed")
            output_subdir = save_path / f"{spk_embed_tag}_{split}"

            # The toolkit / model / device are read from training_config.xvector
            # by the provider itself, on the driver and inside each worker.
            provider = XVectorProvider(
                config=self.training_config,
                params={
                    "manifest_path": str(manifest_path),
                    "output_dir": str(output_subdir),
                },
            )

            runner = XVectorRunner(
                provider=provider,
                batch_size=batch_size,
                async_mode=async_mode,
            )

            logger.info(
                f"Processing {n_utts} utterances for x-vector extraction "
                f"(split: {split})"
            )

            indices = list(range(n_utts))
            results = runner(indices)

            if results is None:
                logger.info(
                    f"Async job submitted for split '{split}'. Check "
                    "result directory for outputs."
                )
                continue

            flat = []
            for item in results:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            n_ok = sum(1 for r in flat if r.get("status") == "ok")
            n_skipped = sum(1 for r in flat if r.get("status") == "skipped")
            logger.info(
                f"X-vectors for split '{split}' saved to {output_subdir} "
                f"({n_ok} new, {n_skipped} skipped)"
            )

        logger.info("X-vector computation completed for all splits")

    def collect_stats(self, *args, **kwargs):
        """Run the collect_stats stage using the GAN-aware trainer.

        Mirrors :meth:`TTSSystem.collect_stats` exactly, except that the
        trainer comes from this module's :func:`_build_trainer`, so a GAN-TTS
        model collects its statistics through ``GANTTSLightningModule`` (which
        also exposes ``speech_shape`` / ``text_shape`` for the ``numel`` batch
        sampler). Like the parent, it builds the trainer without popping
        ``model.normalize`` / ``model.normalize_conf``; see the parent's Notes
        for why that is load-bearing.

        Args:
            *args: Must be empty. Passing any positional argument raises
                ``TypeError`` via ``_reject_stage_args``.
            **kwargs: Must be empty. Passing any keyword argument raises
                ``TypeError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            TypeError: If any positional or keyword arguments are passed.

        Examples:
            ```bash
            python run.py --stages collect_stats \
                --training_config conf/training.yaml
            ```
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        trainer = _build_trainer(self.training_config)
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            getattr(self.training_config, "stats_dir", None),
        )

    def train(self, *args, **kwargs):
        """Run the training stage using the GAN-aware trainer.

        Mirrors :func:`espnet3.systems.base.training.train` exactly, except
        that the trainer comes from this module's :func:`_build_trainer`. The
        override exists only for that dispatch: the base implementation
        resolves ``_build_trainer`` in its own module namespace, so a GAN-TTS
        model would otherwise be wrapped in the plain single-optimizer trainer
        and fail at the first discriminator step.

        Args:
            *args: Must be empty. Passing any positional argument raises
                ``TypeError`` via ``_reject_stage_args``.
            **kwargs: Must be empty. Passing any keyword argument raises
                ``TypeError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            TypeError: If any positional or keyword arguments are passed.

        Notes:
            ``training_config.fit`` is forwarded verbatim to ``trainer.fit``.
            When ``training_config.task`` is set, the espnet2-style
            ``config.yaml`` is written to ``exp_dir`` first so that
            ``Text2Speech`` can rebuild the model at inference time.

        Examples:
            ```bash
            python run.py --stages train --training_config conf/training.yaml
            ```
        """
        self._reject_stage_args("train", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        task = self.training_config.get("task")
        if task:
            save_espnet_config(task, self.training_config, self.training_config.exp_dir)

        trainer = _build_trainer(self.training_config)

        fit_kwargs: Dict[str, Any] = {}
        if hasattr(self.training_config, "fit") and self.training_config.fit:
            fit_kwargs = OmegaConf.to_container(self.training_config.fit, resolve=True)

        trainer.fit(**fit_kwargs)
        logger.info(
            "Training finished in %.2fs | exp_dir=%s model=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            (
                self.training_config.model.get("_target_", None)
                if isinstance(self.training_config.model, DictConfig)
                else None
            ),
        )
