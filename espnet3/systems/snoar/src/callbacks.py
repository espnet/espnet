"""Lightning callback that advances the SNOAR teacher."""

from __future__ import annotations

import lightning.pytorch as pl


class TeacherEMACallback(pl.Callback):
    """Call ``SNOARModel.update_teacher`` once after every optimizer step.

    The step is detected through ``trainer.global_step``, so gradient
    accumulation and skipped (non-finite) batches do not move the teacher.
    The SNOAR training configs add it under ``trainer.callbacks``:

    .. code-block:: yaml

        trainer:
          callbacks:
            - _target_: espnet3.systems.snoar.src.callbacks.TeacherEMACallback
    """

    def __init__(self) -> None:
        """Initialize the step tracker."""
        super().__init__()
        self._last_step = None

    def on_train_start(self, trainer, pl_module):
        """Remember the step training (re)starts from."""
        self._last_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update the teacher when an optimizer step has just been taken."""
        step = trainer.global_step
        if self._last_step is None:
            self._last_step = step
        if step == self._last_step:
            return
        model = getattr(pl_module, "model", pl_module)
        if not hasattr(model, "update_teacher"):
            raise TypeError(
                "TeacherEMACallback needs a model with update_teacher() "
                "(espnet3.systems.snoar.src.model.SNOARModel)"
            )
        for _ in range(step - self._last_step):
            model.update_teacher()
        self._last_step = step
