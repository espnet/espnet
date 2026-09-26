"""Collect LID speech shapes and category mappings with ESPnet3 runners."""

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.parallel.base_runner import BaseRunner, concatenate_shard_files
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel


class LIDCollectStatsProvider(EnvironmentProvider):
    """Build an unprocessed split on each worker without loading a model.

    Args:
        config: Dataset organizer configuration and the ``train`` or ``valid`` mode.

    Example:
        >>> provider = LIDCollectStatsProvider(config)
        >>> dataset = provider.build_env_local()["dataset"]
    """

    def build_env_local(self):
        """Return the split and source-dataset boundaries for global sample IDs."""
        dataset = getattr(instantiate(self.config.dataset), self.config.mode)
        boundaries = []
        offset = 0
        for source in getattr(dataset, "datasets", [dataset]):
            offset += len(source)
            boundaries.append(offset)
        return {"dataset": dataset, "boundaries": boundaries}

    def build_worker_setup_fn(self):
        """Build the same lightweight environment on each local or HPC worker."""
        return self.build_env_local


class LIDCollectStatsRunner(BaseRunner):
    """Write shard-local shapes and merge mappings using global Dataset indices.

    Args:
        provider: Provider for one unprocessed Dataset split.
        output_dir: Statistics root containing train and valid outputs.
        mode: Split name, ``train`` or ``valid``.

    Example:
        >>> runner = LIDCollectStatsRunner(provider, "exp/stats", "train")
        >>> runner(range(len(dataset)))
    """

    def __init__(self, provider, output_dir, mode):
        """Keep raw LID shards separate from optional feature-statistics shards."""
        super().__init__(
            provider,
            batch_size=4,
            output_dir=output_dir,
            shard_subdir=f".lid_shapes/{mode}",
            resume=False,
        )
        self.mode = mode

    @staticmethod
    def forward(indices, dataset, boundaries, **env):
        """Return IDs, waveform shapes, languages and source indices for a batch."""
        rows = []
        for index in indices:
            sample = dataset[index]
            language = sample["lid_labels"]
            if (
                not isinstance(language, str)
                or not language
                or len(language.split()) != 1
            ):
                raise ValueError(
                    "LID statistics require a single language string per sample"
                )
            shape = ",".join(map(str, sample["speech"].shape))
            rows.append((index, shape, language, bisect_right(boundaries, index)))
        return rows

    @staticmethod
    def open_writers(shard_dir, **env):
        """Open separate files so only one worker writes each shard."""
        return {
            name: (shard_dir / name).open("w", encoding="utf-8")
            for name in ("speech_shape", "utt2lang", "utt2dataset")
        }

    @staticmethod
    def write_record(writers, result, state, **env):
        """Write a batch without retaining waveform data in runner state."""
        for index, shape, language, source in result:
            writers["speech_shape"].write(f"{index} {shape}\n")
            writers["utt2lang"].write(f"{index} {language}\n")
            writers["utt2dataset"].write(f"{index} {source}\n")

    def merge(self, shard_dirs):
        """Merge in shard order, retaining the original global sample IDs."""
        destination = self.output_dir / self.mode
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("speech_shape", "utt2dataset"):
            concatenate_shard_files(shard_dirs, name, destination / name)
        for source, targets in (
            ("utt2lang", ("lang2utt", "category2utt")),
            ("utt2dataset", ("dataset2utt",)),
        ):
            groups = defaultdict(list)
            for shard in shard_dirs:
                with (shard / source).open(encoding="utf-8") as stream:
                    for line in stream:
                        index, group = line.split()
                        groups[group].append(index)
            for target in targets:
                with (destination / target).open("w", encoding="utf-8") as stream:
                    for group in sorted(
                        groups, key=int if source == "utt2dataset" else str
                    ):
                        stream.write(f"{group} {' '.join(groups[group])}\n")
        (destination / "batch_keys").write_text("speech\n", encoding="utf-8")
        (destination / "stats_keys").write_text("\n", encoding="utf-8")


def collect_speech_shapes(config) -> None:
    """Collect raw speech shapes and language mappings without a model.

    Args:
        config: Training configuration with ``dataset``, ``stats_dir`` and optional
            ``parallel`` settings. Preprocessing is disabled during collection.

    Returns:
        None. Writes speech_shape, lang2utt, category2utt, dataset2utt,
        utt2dataset, batch_keys and stats_keys under stats_dir/{train,valid}.
        IDs match the CombinedDataset, including speed-perturbed variants.

    Example:
        >>> collect_speech_shapes(training_config)
    """
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.dataset, resolve=True)
    )
    dataset_config.preprocessor = None
    set_parallel(config.get("parallel"))
    providers = {
        mode: LIDCollectStatsProvider(
            OmegaConf.create({"dataset": dataset_config, "mode": mode})
        )
        for mode in ("train", "valid")
    }
    lengths = {
        mode: len(provider.build_env_local()["dataset"])
        for mode, provider in providers.items()
    }
    for mode, length in lengths.items():
        if length == 0:
            raise ValueError(f"Cannot collect speech shapes: {mode} dataset is empty")
    for mode, provider in providers.items():
        LIDCollectStatsRunner(provider, Path(config.stats_dir), mode)(
            range(lengths[mode])
        )
