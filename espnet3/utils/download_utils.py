"""Small helpers for downloading and extracting archives with progress logging."""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    This helper configures a named logger with a standard formatter and:
      - A stream handler for console output (added once per logger).
      - A file handler that writes to `download.log` in `log_dir`, if provided.

    Args:
        name (str): Logger name (e.g., "espnet3.download").
        log_dir (Path | None): Directory for the optional log file.
            When provided, `download.log` is created under this directory.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = (log_dir / "download.log").resolve()
        existing_files = [
            Path(h.baseFilename).resolve()
            for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None)
        ]
        if target not in existing_files:
            fh = logging.FileHandler(target)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def _log(logger: Optional[logging.Logger], message: str) -> None:
    """Log a message using the logger if provided; otherwise print."""
    if logger is None:
        print(message)
    else:
        logger.info(message)


@dataclass
class DownloadProgress:
    """Callable progress hook for urllib to log download percentage."""

    logger: Optional[logging.Logger]
    name: str
    step_percent: int = 5

    def __post_init__(self) -> None:
        """Initialize the internal progress bucket."""
        self._last_bucket: int | None = None

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        """Log progress for the current urllib download callback."""
        if total_size <= 0:
            return

        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)

        bucket = percent // self.step_percent
        if self._last_bucket is None or bucket != self._last_bucket:
            self._last_bucket = bucket
            _log(
                self.logger,
                (
                    f"Downloading {self.name}: {percent}% "
                    f"({downloaded / 1024 / 1024:.1f}MB / "
                    f"{total_size / 1024 / 1024:.1f}MB)"
                ),
            )


def download_url(
    url: str,
    dst_path: Path,
    logger: logging.Logger | None = None,
    step_percent: int = 5,
) -> None:
    """Download a URL to a local path with progress logging.

    This uses `urllib.request.urlretrieve` and logs progress at fixed
    percentage intervals (default: every 5%). The destination directory
    is created if needed.

    Args:
        url (str): The URL to download.
        dst_path (Path): Destination file path.
        logger (logging.Logger | None): Logger to emit progress messages.
            If None, messages are printed to stdout.
        step_percent (int): Percentage step for progress logging.

    Raises:
        URLError: If the download fails.
        HTTPError: If the server returns an error response.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    progress = DownloadProgress(
        logger=logger,
        name=dst_path.name,
        step_percent=step_percent,
    )

    _log(logger, f"Start download: {dst_path.name}")
    _log(logger, f"Target directory: {dst_path.parent.resolve()}")
    urllib.request.urlretrieve(url, dst_path, reporthook=progress)
    _log(logger, f"Download completed: {dst_path.name}")


def extract_targz(
    archive_path: Path,
    dst_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Extract a `.tar.gz` archive into a destination directory.

    Args:
        archive_path (Path): Path to the `.tar.gz` archive.
        dst_dir (Path): Directory to extract files into.
        logger (logging.Logger | None): Logger to emit progress messages.

    Raises:
        tarfile.TarError: If the archive is invalid or extraction fails.
    """
    _log(logger, f"Extracting: {archive_path.name}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)
