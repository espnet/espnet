"""The ESPnet2 S2T task this system trains under.

A subclass of ``espnet2.tasks.s2t.S2TTask`` rather than a copy: the system
needs no change to the task, so it keeps every espnet2 fix and the training
config can still name a task path owned by this system.
"""

from espnet2.tasks.s2t import S2TTask as _S2TTask


class S2TTask(_S2TTask):
    """espnet2's S2T task, unchanged."""
