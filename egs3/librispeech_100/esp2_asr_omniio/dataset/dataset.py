"""LibriSpeech 100h dataset: rows selected by SQL from the omniio dump."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from egs3.librispeech_100.esp2_asr_omniio.dataset.builder import resolve_dump_root
from espnet3.components.data.omniio_dataset import OmniIODataset
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
DEFAULT_QUERY = str(_CONFIG["dataset"]["default_query"])


class LibriSpeech100OmniIODataset(OmniIODataset):
    """ASR items (``speech``, ``text``) selected by a DuckDB query.

    Args:
        query: DuckDB query over the omniio dump; see
            :class:`~espnet3.components.data.omniio_dataset.OmniIODataset`
            for the columns it must return. Defaults to ``default_query`` in
            ``dataset/config.yaml``, one split joined on utterance ID.
        params: Values for the query's ``{name}`` placeholders, e.g.
            ``{"split": "dev-clean"}``.
        dump_root: omniio dump root for ``{dump_root}``; defaults to
            ``$OMNIIO_DUMP_ROOT``, then ``<recipe_dir>/dump/omniio``.
        recipe_dir: Recipe root, used only for that last fallback.
        total_shards: Shards the rows are split into (normally the world
            size); each GPU then loads only its shard's index.
        dist_world_size: Distributed world size; defaults to ``total_shards``.
        shard_by: Column whose hash picks each row's shard.

    Examples:
        >>> ds = LibriSpeech100OmniIODataset(
        ...     params={"split": "dev-clean"}
        ... )  # doctest: +SKIP
        >>> sorted(ds[0])  # doctest: +SKIP
        ['speech', 'text']
    """

    def __init__(
        self,
        query: str | None = None,
        params: Mapping[str, Any] | None = None,
        dump_root: str | Path | None = None,
        recipe_dir: str | Path | None = None,
        total_shards: int | None = None,
        dist_world_size: int | None = None,
        shard_by: str = "id",
    ) -> None:
        """Run ``query`` (or the default one) against the dump."""
        super().__init__(
            query=query or DEFAULT_QUERY,
            fields={"speech": "audio", "text": "text"},
            params=params,
            dump_root=resolve_dump_root(recipe_dir, dump_root),
            total_shards=total_shards,
            dist_world_size=dist_world_size,
            shard_by=shard_by,
        )
