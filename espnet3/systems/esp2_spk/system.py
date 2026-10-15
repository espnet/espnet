"""Speaker verification system on the ESPnet2 speaker task."""

from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class SpeakerSystem(BaseSystem):
    """Speaker verification system.

    Trains on fixed-length raw waveform crops, so it needs neither a
    tokenizer nor feature statistics: ``collect_stats`` is not a stage. Nor
    is there a demo yet, so ``pack_demo``/``upload_demo`` are left out.

    Examples:
        >>> [s.name for s in SpeakerSystem.stages]
        ['create_dataset', 'train', 'infer', 'measure', 'pack_model', 'upload_model']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
    )
