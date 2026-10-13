"""LID system implementation."""

from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.esp2_lid.src.collect_stats import collect_speech_shapes


class LIDSystem(BaseSystem):
    """Language identification system.

    The stages are the base system's, without the demo stages: there is no
    LID demo yet. ``collect_stats`` first writes the speech shapes and the
    language mappings the category samplers read.

    Examples:
        >>> [s.name for s in LIDSystem.stages]
        ['create_dataset', 'collect_stats', 'train', 'infer', 'measure', \
'pack_model', 'upload_model']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="collect_stats", config="training", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
    )

    def collect_stats(self, *args, **kwargs):
        """Collect LID metadata and optionally feature statistics.

        Writes ``speech_shape``, ``lang2utt``, ``category2utt``,
        ``dataset2utt`` and ``utt2dataset`` under ``stats_dir/{train,valid}``
        (see :func:`collect_speech_shapes`). When
        ``model.model_conf.extract_feats_in_collect_stats`` is true or unset,
        it then collects feature statistics as the base system does.
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        collect_speech_shapes(self.training_config)
        model_conf = self.training_config.model.get("model_conf", {})
        if model_conf.get("extract_feats_in_collect_stats", True):
            return super().collect_stats()
