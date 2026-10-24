"""Parallel download and ZIP extraction for the VoxLingua107 recipe."""

import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.utils.download_utils import download_url


class DownloadProvider(EnvironmentProvider):
    """Supply serializable file preparation tasks to local or HPC workers.

    Args:
        tasks: List of dictionaries consumed by the preparation runner.

    Example:
        >>> provider = DownloadProvider([{"url": url, "path": "download/en.zip"}])
    """

    def __init__(self, tasks):
        """Store paths as strings so shard specifications are portable."""
        super().__init__(OmegaConf.create({"tasks": tasks}))

    def build_env_local(self):
        """Return the task list for indexed processing."""
        return {"dataset": OmegaConf.to_container(self.config.tasks, resolve=True)}

    def build_worker_setup_fn(self):
        """Create the same task list on each worker."""
        return self.build_env_local


class DownloadRunner(BaseRunner):
    """Download one file per task and optionally extract a ZIP.

    Task fields are ``url``, ``path`` and optional ``extract_to``. Existing files
    are reused. Fresh transfers use the common download helper; interrupted
    ``.part`` files resume with wget. ZIP extraction retains unzip's CRC checks.

    Example:
        >>> DownloadRunner(provider, output_dir="download/.jobs", resume=False)([0])
    """

    @staticmethod
    def forward(index, dataset, **env):
        """Download one task and finish extraction before marking its shard done."""
        task = dataset[index]
        path = Path(task["path"])
        if not path.is_file():
            temporary = path.with_suffix(path.suffix + ".part")
            if temporary.exists():
                subprocess.run(
                    ["wget", "--continue", "-O", str(temporary), task["url"]],
                    check=True,
                )
            else:
                download_url(task["url"], temporary)
            temporary.replace(path)
        if task.get("extract_to"):
            destination = Path(task["extract_to"])
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["unzip", "-q", "-o", str(path), "-d", str(destination)], check=True
            )
