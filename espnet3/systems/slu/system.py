"""SLU system implementation.

This module provides the SLU system on top of the ASR system. SLU-specific
logic (postdecoder, deliberation encoder, dual tokenization) is handled by
the Task class and SLUPreprocessor, so the system itself is a pass-through.
"""

from espnet3.systems.asr.system import ASRSystem


class SLUSystem(ASRSystem):
    """SLU-specific system.

    Inherits all ASR system stages (create_dataset, train, train_tokenizer,
    infer, metric). SLU-specific model construction is handled by
    :class:`espnet3.systems.slu.task.SLUTask`.
    """

    pass
