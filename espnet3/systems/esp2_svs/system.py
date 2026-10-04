"""SVS system implementation.

Singing voice synthesis reuses the TTS data stages (long/short utterance
removal and token-list creation) and adds GAN-aware training for the
``espnet2.tasks.gan_svs`` models.
"""

import logging

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import train
from espnet3.systems.esp2_svs.gan_lightning_module import GANLightningModule
from espnet3.systems.tts.system import TTSSystem

logger = logging.getLogger(__name__)


class SVSSystem(TTSSystem):
    """SVS-specific system.

    The data stages come from :class:`~espnet3.systems.tts.system.TTSSystem`.
    Two stages differ from it:

    - ``collect_stats`` is the base-system stage, which drops
      ``model.normalize`` while collecting. SVS configs normalize with
      ``global_mvn``, whose stats file is what this stage writes.
    - ``train`` wraps the model in
      :class:`~espnet3.systems.esp2_svs.gan_lightning_module.GANLightningModule`.
    """

    def collect_stats(self, *args, **kwargs):
        """Collect feature statistics with the base-system stage."""
        return BaseSystem.collect_stats(self, *args, **kwargs)

    def train(self, *args, **kwargs):
        """Train the model, in generator/discriminator turns for GAN models."""
        self._reject_stage_args("train", args, kwargs)
        logger.info(
            "Training start | exp_dir=%s task=%s",
            getattr(self.training_config, "exp_dir", None),
            getattr(self.training_config, "task", None),
        )
        return train(self.training_config, module_cls=GANLightningModule)
