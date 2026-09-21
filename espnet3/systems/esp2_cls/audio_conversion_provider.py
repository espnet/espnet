"""Provider for parallel audio format conversion."""

import logging
import shutil
from typing import Any, Callable, Dict, List, Tuple

from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)


class AudioConversionProvider(EnvironmentProvider):
    """Provider for converting source clips to the recipe's audio format.

    Builds the job list and the ``ffmpeg`` invocation shared by
    :class:`espnet3.systems.esp2_cls.audio_conversion_runner.AudioConversionRunner`
    workers. Nothing expensive is loaded here: a worker only needs the path to
    ``ffmpeg`` and the target format.

    A recipe's ``DatasetBuilder.build`` collects the conversions it needs and
    hands them over as ``jobs``:

    .. code-block:: python

        provider = AudioConversionProvider(
            config=training_config,
            params={
                "jobs": [(clip, wav) for clip, wav in pending],
                "sampling_rate": 16000,
            },
        )
        runner = AudioConversionRunner(provider=provider, output_dir=shards)
        results = runner(range(len(pending)))
    """

    def __init__(self, config: DictConfig, params: Dict[str, Any] | None = None):
        """Initialize the provider.

        Args:
            config: Configuration of the stage that owns the conversion.
            params: Extra parameters forwarded from the driver to workers:
                ``jobs`` (required), ``sampling_rate`` (required),
                ``channels`` (optional, defaults to 1) and ``ffmpeg``
                (optional, defaults to whatever is on ``PATH``).
        """
        super().__init__(config)
        self.params = params or {}

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local execution.

        Returns:
            Dict[str, Any]: The job list and conversion settings consumed by
            ``AudioConversionRunner.forward``.

        Raises:
            RuntimeError: If a required parameter is missing or ``ffmpeg`` is
                not installed.

        Example:
            >>> provider = AudioConversionProvider(
            ...     config=OmegaConf.create({}),
            ...     params={"jobs": [("a.mp4", "a.wav")], "sampling_rate": 16000},
            ... )
            >>> sorted(provider.build_env_local())
            ['channels', 'ffmpeg', 'jobs', 'sampling_rate']
        """
        return AudioConversionProvider._build_env(self.params)

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create a worker setup function for distributed execution.

        Returns:
            Callable[[], Dict[str, Any]]: A zero-arg callable executed once
            per worker that returns the environment dictionary consumed by
            ``AudioConversionRunner.forward``.
        """
        params = self.params

        def setup() -> Dict[str, Any]:
            return AudioConversionProvider._build_env(params)

        return setup

    @staticmethod
    def _build_env(params: Dict[str, Any]) -> Dict[str, Any]:
        jobs: List[Tuple[str, str]] = params.get("jobs", None)
        if jobs is None:
            raise RuntimeError("jobs must be provided for audio conversion.")

        sampling_rate = params.get("sampling_rate", None)
        if sampling_rate is None:
            raise RuntimeError("sampling_rate must be provided for audio conversion.")

        # Resolved on the driver and on every worker, so a node without ffmpeg
        # fails during setup rather than once per clip.
        ffmpeg = params.get("ffmpeg") or shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found in PATH. Install it or pass its path as the "
                "`ffmpeg` parameter."
            )

        return {
            "jobs": [(str(src), str(dst)) for src, dst in jobs],
            "sampling_rate": int(sampling_rate),
            "channels": int(params.get("channels", 1)),
            "ffmpeg": str(ffmpeg),
        }
