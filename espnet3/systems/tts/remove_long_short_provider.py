"""Provider for parallel long/short utterance duration filtering."""

import logging
from typing import Any, Callable, Dict, List, Tuple

from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)


class RemoveLongShortProvider(EnvironmentProvider):
    """Provider for the ``remove_long_short`` stage.

    Builds the manifest entries and duration bounds shared by
    ``RemoveLongShortRunner`` workers. No model needs to be loaded here since
    duration filtering only reads WAV headers.
    """

    def __init__(self, config: DictConfig, params: Dict[str, Any] | None = None):
        """Initialize the provider.

        Args:
            config: Configuration with remove_long_short settings.
            params: Extra parameters (``manifest_path``, ``min_duration``,
                ``max_duration``) forwarded from the driver to workers.
        """
        super().__init__(config)
        self.params = params or {}

    def build_env_local(self) -> Dict[str, Any]:
        """Build environment once on driver for local execution.

        Returns:
            A dictionary containing the manifest entries and duration bounds
            needed by ``RemoveLongShortRunner.forward``.

        Raises:
            RuntimeError: If required parameters are missing.
        """
        return RemoveLongShortProvider._build_env(self.params)

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create a worker setup function for distributed execution.

        Returns:
            A zero-arg callable executed once per worker that returns the
            environment dictionary consumed by ``RemoveLongShortRunner.forward``.
        """
        params = self.params

        def setup() -> Dict[str, Any]:
            return RemoveLongShortProvider._build_env(params)

        return setup

    @staticmethod
    def _build_env(params: Dict[str, Any]) -> Dict[str, Any]:
        manifest_path = params.get("manifest_path", None)
        if manifest_path is None:
            raise RuntimeError(
                "Please provide manifest_path obtained from create_dataset stage"
            )

        min_duration = params.get("min_duration", None)
        max_duration = params.get("max_duration", None)
        if min_duration is None or max_duration is None:
            raise RuntimeError(
                "min_duration and max_duration must be provided for "
                "remove_long_short stage."
            )

        entries, n_dropped_empty = RemoveLongShortProvider._load_entries(manifest_path)

        return {
            "entries": entries,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "n_dropped_empty": n_dropped_empty,
        }

    @staticmethod
    def _load_entries(manifest_path) -> Tuple[List[Tuple[str, str, str]], int]:
        r"""Parse a TSV manifest into kept entries + count of empty-text drops.

        Mirrors espnet2's ``NF != 1`` filter: rows without text are dropped
        here so they never reach duration filtering.

        Each line is expected to be ``utt_id\twav_path\ttext\tspeaker_id``.
        Blank lines are skipped.

        Returns:
            A tuple of (entries, n_dropped_empty) where entries is a list of
            ``(utt_id, wav_path, line)`` tuples in original manifest order,
            and ``line`` is newline-terminated for direct write-back.
        """
        entries: List[Tuple[str, str, str]] = []
        n_dropped_empty = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped:
                    continue
                parts = stripped.split("\t")
                if len(parts) < 3 or parts[2].strip() == "":
                    n_dropped_empty += 1
                    continue
                utt_id, wav_path = parts[0], parts[1]
                entries.append(
                    (utt_id, wav_path, line if line.endswith("\n") else line + "\n")
                )
        return entries, n_dropped_empty
