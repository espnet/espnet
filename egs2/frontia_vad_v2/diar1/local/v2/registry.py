"""Registry of the corpora: directory, domain, label source, licence, reader.

Domains (the categories balanced by --chunk_upsampling_factor):
    meeting, conversation, farfield, broadcast, read, children,
    vocal_nonspeech, music, environment, animal, quiet_closemic
"""

from dataclasses import dataclass
from typing import Callable, Dict

DOMAINS = (
    "meeting",
    "conversation",
    "farfield",
    "broadcast",
    "read",
    "children",
    "vocal_nonspeech",
    "music",
    "environment",
    "animal",
    "quiet_closemic",
)


@dataclass(frozen=True)
class Adapter:
    directory: str  # under the corpora root ("" if none)
    domain: str
    label_source: str
    licence: str
    prepare: Callable  # (corpus dir, Writers, shard, nshards) -> None
    # Writers[split] is the VadDumpWriter of split "train" or "dev"


ADAPTERS: Dict[str, Adapter] = {}


def corpus(name: str, directory: str, domain: str, label_source: str, licence: str):
    """Register the decorated reader as corpus `name`."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain}")

    def register(prepare):
        ADAPTERS[name] = Adapter(directory, domain, label_source, licence, prepare)
        return prepare

    return register


def mine(index: int, shard: int, nshards: int) -> bool:
    """Whether item `index` belongs to this shard."""
    return index % nshards == shard


AUTO = "auto:"  # prefix of a split decided by group (split_by_hash)


def group_hash(key: str) -> int:
    """A stable hash of a group key."""
    import hashlib

    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def hash_split(key: str, dev_percent: int = 2) -> str:
    """Return "dev" for about dev_percent % of group keys, else "train"."""
    return "dev" if group_hash(key) % 100 < dev_percent else "train"


def split_by_hash(key: str) -> str:
    """The split of a recording of group `key`, for corpora without a dev split.

    The key groups what must not be split, e.g. the speaker, session or
    meeting. Writers[split_by_hash(key)] writes into hash_split(key) and
    records the group in the manifest; stage 5 then fixes the dev groups of
    the corpus with dev_groups() over all its groups (local/v2/splits.py), so
    every such corpus has dev data and no group is in both splits.
    """
    return AUTO + key


def dev_groups(groups, dev_percent: int = 2, few: int = 50) -> set:
    """The dev groups of a corpus, from all its group keys.

    With `few` groups or fewer, the one group of lowest hash; else the groups
    hash_split() puts in dev (about dev_percent %), or the group of lowest
    hash if none.
    """
    groups = sorted(set(groups))
    if not groups:
        return set()
    lowest = min(groups, key=lambda g: (group_hash(g), g))
    if len(groups) <= few:
        return {lowest}
    dev = {g for g in groups if hash_split(g, dev_percent) == "dev"}
    return dev or {lowest}
