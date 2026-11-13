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


def split_by_hash(key: str, dev_percent: int = 2) -> str:
    """ "dev" for about dev_percent % of keys, else "train"; stable across runs.

    Use a key that groups what must not be split, e.g. the speaker or the
    session, for corpora without an official dev split.
    """
    import hashlib

    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return "dev" if h % 100 < dev_percent else "train"
