"""S2T Task.

Note: This file re-exports the corresponding espnet2 task class, which the
system instantiates. esp2_asr keeps a copy of its task here because it adds
a transducer variant beside it; s2t needs no changes, so importing keeps the
two in step rather than letting a copy drift.
"""

from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.s2t_ctc import S2TTask as S2TCTCTask

__all__ = ["S2TCTCTask", "S2TTask"]
