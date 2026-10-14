"""The interfaces of ESPnet3: what every system declares, in one place.

A system under ``espnet3/systems/<name>/`` may organise its training however
it likes; what it exposes to the rest of the toolkit is declared here, so a
front end written against an interface works for every system that
implements it.

Every interface is declared the same way, as tuples of :class:`Field`
whose ``kind`` is a :class:`Kind` registered in :data:`KINDS`:

- :mod:`espnet3.api.field` - one declared input or output.
- :mod:`espnet3.api.kinds` - what a field can hold, and :func:`register_kind`
  for a new modality.
- :mod:`espnet3.api.check` - :func:`check_declaration`, the rules every
  declaration follows, whatever it declares.

and the interfaces themselves:

- :mod:`espnet3.api.inference` - calling a trained model: the fields it
  takes and returns, how audio reaches it, and how the ``infer`` stage's
  provider and runner run it.

The candidates for what comes next, in the order they would pay off, are
the item a recipe's ``dataset/`` must return, the ``metric(...)`` call the
``measure`` stage makes, and the trainable model that fine-tuning loads
weights into. Each is a de facto convention today; it belongs here once it
is written down.

This package says what a part declares. Checking a part against its
declaration at run time - the values it is given, what it returns, how
its config binds it to other parts - is :mod:`espnet3.contract`, which
depends on this package and never the reverse.
"""

from espnet3.api.check import check_declaration
from espnet3.api.field import Field
from espnet3.api.kinds import KINDS, Kind, register_kind

__all__ = [
    "KINDS",
    "Field",
    "Kind",
    "check_declaration",
    "register_kind",
]
