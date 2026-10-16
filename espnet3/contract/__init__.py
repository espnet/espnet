"""Contract declarations and checks for ESPnet3 components.

``Field`` and its registered ``Kind``s describe inputs, outputs and data
fields. :mod:`.data`, :mod:`.metrics` and :mod:`.stages` enforce those
declarations at their respective boundaries.
"""

from espnet3.contract.check import check_declaration
from espnet3.contract.field import Field
from espnet3.contract.kinds import KINDS, Kind, register_kind

__all__ = ["KINDS", "Field", "Kind", "check_declaration", "register_kind"]
