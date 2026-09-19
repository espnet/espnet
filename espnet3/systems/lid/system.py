"""LID system implementation."""

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.lid.collect_stats import collect_speech_shapes


class LIDSystem(BaseSystem):
    """Language identification system."""

    def collect_stats(self, *args, **kwargs):
        """Collect LID metadata and optionally feature statistics."""
        self._reject_stage_args("collect_stats", args, kwargs)
        collect_speech_shapes(self.training_config)
        model_conf = self.training_config.model.get("model_conf", {})
        if model_conf.get("extract_feats_in_collect_stats", True):
            return super().collect_stats()
