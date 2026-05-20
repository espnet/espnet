"""FalAR dataset builder."""

from __future__ import annotations

import json
import logging
import re
from importlib import resources
from pathlib import Path

from datasets import load_dataset_builder
from datasets.download import DownloadConfig, DownloadManager
from huggingface_hub import HfApi, hf_hub_url
from omegaconf import DictConfig, OmegaConf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_recipe_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_recipe_config()
_BUILDER_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]
_HF_DATASET = str(_BUILDER_CFG["hf_dataset"])
_STATE_PATH = str(_BUILDER_CFG["prepare_state_path"])
_TRAIN_SHARDS = [str(split) for split in _DATASET_CFG["train_shards"]]
_DOWNLOAD_SPLITS = [
    *_TRAIN_SHARDS,
    str(_DATASET_CFG["dev_split"]),
    str(_DATASET_CFG["test_split"]),
]
_PARQUET_REVISION = "refs/convert/parquet"
_PARQUET_SPLIT_RE = re.compile(
    r"^(?P<split>.+)-\d{5}-of-\d{5}.*\.parquet$"
)


def _cache_dir_as_str(cache_dir: str | Path | None) -> str | None:
    if cache_dir is None:
        return None
    return str(Path(cache_dir))


def _download_file(
    item: dict[str, str],
    cache_dir: str | None = None,
) -> dict[str, str]:
    manager = DownloadManager(
        download_config=DownloadConfig(cache_dir=cache_dir),
    )
    cached_path = manager.download(item["url"])
    return {
        "split": item["split"],
        "url": item["url"],
        "cached_path": str(cached_path),
    }


def _iter_builder_data_files(cache_dir: str | None) -> list[dict[str, str]]:
    builder = load_dataset_builder(
        _HF_DATASET,
        cache_dir=cache_dir,
    )
    data_files = getattr(builder.config, "data_files", None)
    if not data_files:
        return []

    items: list[dict[str, str]] = []
    for split_name, files in data_files.items():
        if split_name not in _DOWNLOAD_SPLITS:
            continue
        for file_url in files:
            file_url = str(file_url)
            if not file_url.endswith(".parquet"):
                continue
            items.append({"split": str(split_name), "url": file_url})
    return items


def _iter_repo_parquet_files() -> list[dict[str, str]]:
    api = HfApi()
    items: list[dict[str, str]] = []
    for repo_file in api.list_repo_files(
        repo_id=_HF_DATASET,
        repo_type="dataset",
        revision=_PARQUET_REVISION,
    ):
        if not repo_file.endswith(".parquet"):
            continue
        file_name = Path(repo_file).name
        match = _PARQUET_SPLIT_RE.match(file_name)
        if match is None:
            continue
        split_name = match.group("split")
        if split_name not in _DOWNLOAD_SPLITS:
            continue
        items.append(
            {
                "split": split_name,
                "url": hf_hub_url(
                    repo_id=_HF_DATASET,
                    filename=repo_file,
                    repo_type="dataset",
                    revision=_PARQUET_REVISION,
                ),
            }
        )
    return items


def _resolve_download_items(cache_dir: str | None) -> list[dict[str, str]]:
    items = _iter_builder_data_files(cache_dir)
    if items:
        return items
    return _iter_repo_parquet_files()


class FalarDownloadProvider(EnvironmentProvider):
    """Provider for FalAR cache warm-up jobs."""

    def __init__(self, cache_dir: str | None) -> None:
        super().__init__(OmegaConf.create({"cache_dir": cache_dir}))

    def build_env_local(self) -> dict:
        return {"cache_dir": self.config.get("cache_dir")}

    def build_worker_setup_fn(self):
        cache_dir = self.config.get("cache_dir")

        def setup():
            return {"cache_dir": cache_dir}

        return setup


class FalarDownloadRunner(BaseRunner):
    """Runner that downloads FalAR parquet files into the Hugging Face cache."""

    @staticmethod
    def forward(item: dict[str, str], cache_dir=None, **_env):
        return _download_file(item, cache_dir=cache_dir)

    def merge(self, shard_dirs: list[Path]) -> list[dict[str, int | str]]:
        results: list[dict[str, int | str]] = []
        for shard_dir in shard_dirs:
            state = self.deserialize_state(shard_dir)
            results.extend(state.get("records", []))
        return results


class FalarBuilder(DatasetBuilder):
    """Prepare Hugging Face cache entries for the FalAR recipe dataset."""

    @staticmethod
    def _state_file(recipe_dir: str | Path) -> Path:
        recipe_root = Path(recipe_dir).resolve()
        return recipe_root / _STATE_PATH

    @staticmethod
    def _normalize_parallel_config(parallel) -> DictConfig | None:
        if parallel is None:
            return None
        if isinstance(parallel, DictConfig):
            return parallel
        return OmegaConf.create(parallel)

    def _validate_config(self) -> None:
        if not _HF_DATASET:
            raise ValueError("FalAR builder config must define builder.hf_dataset.")

    @staticmethod
    def _get_num_workers(parallel) -> int:
        parallel_config = FalarBuilder._normalize_parallel_config(parallel)
        if parallel_config is None:
            return 1
        return max(1, int(parallel_config.get("n_workers", 1)))

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        cache_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether all configured FalAR parquet files were cached before."""
        self._validate_config()
        state_path = self._state_file(recipe_dir)
        if not state_path.is_file():
            return False

        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected_cache_dir = _cache_dir_as_str(cache_dir)
        return (
            state.get("hf_dataset") == _HF_DATASET
            and state.get("cache_dir") == expected_cache_dir
            and state.get("num_files", 0) > 0
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        cache_dir: str | Path | None = None,
        parallel=None,
        **_kwargs,
    ) -> None:
        """Download FalAR parquet files, optionally in parallel."""
        self._validate_config()
        cache_dir_str = _cache_dir_as_str(cache_dir)
        items = _resolve_download_items(cache_dir_str)
        if not items:
            raise RuntimeError("No FalAR parquet files were resolved for download.")
        parallel_config = self._normalize_parallel_config(parallel)
        num_workers = self._get_num_workers(parallel_config)
        if parallel_config is not None and num_workers > 1:
            set_parallel(parallel_config)

        runner = FalarDownloadRunner(
            provider=FalarDownloadProvider(cache_dir=cache_dir_str),
            output_dir=self._state_file(recipe_dir).parent / "_prepare_source",
        )
        results = runner(items)

        state_path = self._state_file(recipe_dir)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "hf_dataset": _HF_DATASET,
                    "cache_dir": cache_dir_str,
                    "num_files": len(items),
                    "splits": sorted({item["split"] for item in items}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Prepared FalAR source cache with %d worker(s) for %d parquet files",
            num_workers,
            len(results),
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Return ``True`` because this recipe has no build artifacts."""
        self._validate_config()
        return True

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """No-op build step for Hugging Face-backed FalAR access."""
        self._validate_config()
