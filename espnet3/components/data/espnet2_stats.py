"""Native ESPnet2 statistics accumulation for ESPnet3 datasets."""

from copy import deepcopy
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from espnet2.bin.aggregate_stats_dirs import aggregate_stats_dirs
from espnet2.main_funcs.collect_stats import collect_stats
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.utils.task_utils import get_espnet_model


class _StatisticsDataset(IterableDataset):
    """Read one contiguous shard with ESPnet2's worker striding."""

    def __init__(self, dataset, start, stop):
        """Retain the dataset and half-open shard bounds."""
        self.dataset = dataset
        self.start = start
        self.stop = stop

    def __iter__(self):
        """Yield this worker's strided items within the shard."""
        worker = get_worker_info()
        offset = worker.id if worker else 0
        step = worker.num_workers if worker else 1
        for index in range(self.start + offset, self.stop, step):
            yield self.dataset[index]


def collect_stats_espnet2(config):
    """Collect CPU statistics using the source recipe's arithmetic order.

    Select with ``espnet2_stats: {nj: 32, batch_size: 20, num_workers: 1}``
    in a training config. The contiguous shards match ``split_scp.pl``;
    native collectors accumulate utterances and then shards in that order.
    This is needed when migrating recipes whose GlobalMVN must reproduce
    ESPnet2 rather than ESPnet3's per-batch reduction. Training augmentation
    is disabled, as in ESPnet2's collect-stats preprocessing.

    Args:
        config: Training DictConfig with task, model, dataset, stats_dir,
            seed and espnet2_stats. The latter requires nj and batch_size;
            num_workers defaults to one. Dataset uses DataOrganizer with
            ``_recursive_: false``. Models are instantiated on CPU with
            normalization disabled without mutating the training config.

    Returns:
        None. Writes native shape files and NPZ statistics to stats_dir,
        retaining per-shard outputs under stats_dir/logdir. Reruns replace
        those files; old artifacts outside the selected shards are ignored.

    Raises:
        ValueError: A split is empty or a batch/shard/worker setting is invalid.
        TypeError: The task is missing or the dataset lacks ESPnet collation.
    """
    settings = config.espnet2_stats
    batch_size = int(settings.batch_size)
    nj = int(settings.nj)
    workers = int(settings.get("num_workers", 1))
    if min(batch_size, nj) < 1 or workers < 0:
        raise ValueError("Positive batch_size/nj and nonnegative num_workers required")
    if not config.get("task"):
        raise TypeError("Native statistics require an ESPnet task")
    organizer = instantiate(config.dataset)
    datasets = [organizer.train, organizer.valid]
    if any(dataset is None or len(dataset) == 0 for dataset in datasets):
        raise ValueError("Native statistics require nonempty train and valid splits")
    nj = min(nj, *(len(dataset) for dataset in datasets))
    for dataset in datasets:
        if not dataset.use_espnet_preprocessor:
            raise TypeError("Native statistics require ESPnet preprocessing")
        dataset.use_espnet_collator = True
        for _, preprocessor in dataset.transforms:
            if isinstance(preprocessor, AbsPreprocessor):
                preprocessor.train = False
    model_config = deepcopy(config.model)
    model_config.normalize = None
    model_config.normalize_conf = {}
    seed = int(config.get("seed", 0))
    set_all_random_seed(seed)
    model = (
        get_espnet_model(
            config.task, OmegaConf.to_container(model_config, resolve=True)
        )
        .cpu()
        .eval()
    )
    output_dir = Path(config.stats_dir)
    shard_dirs = []
    for shard in range(nj):
        set_all_random_seed(seed)
        loaders = []
        for dataset in datasets:
            size, extra = divmod(len(dataset), nj)
            start = shard * size + min(shard, extra)
            stop = start + size + int(shard < extra)
            loaders.append(
                DataLoader(
                    _StatisticsDataset(dataset, start, stop),
                    batch_size=batch_size,
                    num_workers=workers,
                    collate_fn=CommonCollateFn(int_pad_value=-1),
                )
            )
        directory = output_dir / "logdir" / f"stats.{shard + 1}"
        collect_stats(
            model,
            *loaders,
            output_dir=directory,
            ngpu=0,
            log_interval=100,
            write_collected_feats=False,
        )
        shard_dirs.append(directory)
    aggregate_stats_dirs(shard_dirs, output_dir, "INFO", False)
