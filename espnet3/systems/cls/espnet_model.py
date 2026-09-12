"""Classification model for ESPnet3."""

from __future__ import annotations

import logging
from typing import Iterable

from espnet2.cls.espnet_model import ESPnetClassificationModel

logger = logging.getLogger(__name__)


class ClassificationModel(ESPnetClassificationModel):
    """Classification model that honours ``freeze_param``.

    ESPnet2 freezes parameters in ``AbsTask.main_worker()``, which ESPnet3 never
    runs: it reaches the model through ``CLSTask.build_model()`` alone. ESPnet3
    also keeps its stage code out of model internals, so the model applies the
    freeze itself.

    Behaviour is identical to
    :class:`espnet2.cls.espnet_model.ESPnetClassificationModel` when
    ``freeze_param`` is empty.

    Args:
        freeze_param: Parameter or module names to exclude from training, e.g.
            ``["frontend.upstream"]``. A name matches itself and everything
            beneath it.
        *args: Forwarded to the ESPnet2 model.
        **kwargs: Forwarded to the ESPnet2 model.

    Examples:
        Selected from a recipe's ``conf/training.yaml``:

        .. code-block:: yaml

            task: espnet3.systems.cls.task.CLSTask
            model:
              model: espnet          # resolved through `model_choices`
              freeze_param:
                - frontend.upstream

        A frozen WavLM frontend leaves 1.42 M of 95.80 M parameters trainable.
    """

    def __init__(self, *args, freeze_param: Iterable[str] = (), **kwargs):
        """Build the ESPnet2 model, then freeze the configured modules."""
        super().__init__(*args, **kwargs)
        self.freeze_param = [str(name) for name in freeze_param or []]
        for name in self.freeze_param:
            frozen = 0
            for key, param in self.named_parameters():
                if key == name or key.startswith(name + "."):
                    param.requires_grad = False
                    frozen += 1
            if frozen == 0:
                logger.warning("freeze_param matched no parameter: %s", name)
            else:
                logger.info("Froze %d parameter tensor(s) under %s", frozen, name)
