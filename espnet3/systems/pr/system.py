"""Phone recognition system implementation.

This module defines the system used by phone recognition recipes. A phone
recognizer predicts a sequence of phones (normally IPA) rather than words or
subwords, which changes what a recipe needs from the framework: the token
inventory is fixed by the phone set instead of being learned, and the scores of
interest are phone-level. Everything else -- dataset creation, statistics
collection, training, inference and measurement -- is the base pipeline.
"""

import logging

from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class PRSystem(BaseSystem):
    """Phone recognition system.

    This system runs the base stages unchanged: ``create_dataset``,
    ``collect_stats``, ``train``, ``infer`` and ``measure``. It deliberately
    does not extend :class:`espnet3.systems.asr.system.ASRSystem`, whose
    ``train`` trains a SentencePiece tokenizer first. A phone recognizer's
    inventory is the phone set itself, so there is nothing to learn and
    ``train_tokenizer`` is not part of this family's stage list.

    Pair it with ``egs3/TEMPLATE/pr`` for the matching config defaults, and with
    :mod:`espnet3.systems.pr.metrics` for PER and PFER.

    Examples:
        A recipe selects this system from its ``run.py``::

            from egs3.TEMPLATE.pr.run import DEFAULT_STAGES, build_parser, main
            from espnet3.systems.pr.system import PRSystem

            main(args=args, system_cls=PRSystem, stages=DEFAULT_STAGES)
    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
        demo_config: DictConfig | None = None,
    ) -> None:
        """Initialize the phone recognition system with optional stage configs.

        Args:
            training_config: Training configuration. May be ``None`` for a
                recipe that only runs ``infer`` and ``measure`` against a
                pretrained model.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for model packing
                and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides.
            demo_config: Demo configuration for demo packing and upload stages.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping=stage_log_mapping,
            demo_config=demo_config,
        )
