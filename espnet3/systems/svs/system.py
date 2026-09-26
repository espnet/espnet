"""SVS system implementation.

Singing voice synthesis reuses the TTS data stages (long/short utterance
removal and token-list creation) and adds GAN-aware training for the
``espnet2.tasks.gan_svs`` models.
"""

import logging

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import train
from espnet3.systems.svs.gan_lightning_module import GANLightningModule
from espnet3.systems.tts.system import TTSSystem

logger = logging.getLogger(__name__)


class SVSSystem(TTSSystem):
    """SVS-specific system.

    Stages come from :class:`~espnet3.systems.tts.system.TTSSystem`
    (``remove_long_short``, ``create_token_list``) and
    :class:`~espnet3.systems.base.system.BaseSystem`, with two differences:

    - ``collect_stats`` uses the base implementation, which drops
      ``model.normalize`` while the stats are being collected. SVS configs
      set ``normalize: global_mvn`` with a ``stats_file`` this stage is
      about to produce, so the TTS override that keeps ``normalize`` would
      fail here. Set ``write_collected_feats: true`` to also dump the
      features and pitch for the training stage.
    - ``train`` wraps the model in
      :class:`~espnet3.systems.svs.gan_lightning_module.GANLightningModule`,
      which runs the generator and discriminator turns of
      ``ESPnetGANSVSModel``. Non-GAN models are trained as usual.
    """

    def collect_stats(self, *args, **kwargs):
        """Collect feature statistics with the base-system stage."""
        return BaseSystem.collect_stats(self, *args, **kwargs)

    def train(self, *args, **kwargs):
        """Train the SVS model, with GAN turns when the model needs them."""
        self._reject_stage_args("train", args, kwargs)
        logger.info(
            "Training start | exp_dir=%s task=%s",
            getattr(self.training_config, "exp_dir", None),
            getattr(self.training_config, "task", None),
        )
        return train(self.training_config, module_cls=GANLightningModule)
