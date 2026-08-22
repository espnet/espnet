"""Enhancement system for ESPnet3."""

import logging

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class EnhancementSystem(BaseSystem):
    """Speech enhancement system."""

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
            **kwargs,
        )
