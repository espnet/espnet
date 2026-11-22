"""The layout version of a ``pack_model`` bundle.

``pack_model`` writes it to ``meta.yaml``; the loaders refuse a bundle from
a newer one. Kept in its own module so both can import it without a cycle.
"""

PACK_SCHEMA_VERSION = 1
