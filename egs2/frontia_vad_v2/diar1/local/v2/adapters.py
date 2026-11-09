"""All corpus readers, by group; importing a module registers its corpora.

Each adapters_<group>.py registers its corpora with registry.corpus(). A
reader gets the corpus directory, the Writers of its splits (prep.py), and
its shard; it adds every recording of its shard with Writers[split].add().
"""

import adapters_synthetic  # noqa: F401
from registry import ADAPTERS, DOMAINS  # noqa: F401
