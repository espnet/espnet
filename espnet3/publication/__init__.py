"""Publishing an ESPnet3 model: the bundle, and what it is loaded with.

``pack_model`` writes a bundle; :func:`espnet3.api.inference.load` reads
one. :data:`PACK_SCHEMA_VERSION` is the bundle layout both agree on.
"""

from espnet3.publication.schema import PACK_SCHEMA_VERSION

__all__ = ["PACK_SCHEMA_VERSION"]
