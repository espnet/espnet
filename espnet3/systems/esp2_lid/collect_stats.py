"""Statistics collection for LID systems."""

from collections import defaultdict
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from espnet2.main_funcs.collect_stats import collect_stats


def _get_num_workers(config, mode: str) -> int:
    mode_config = getattr(config.dataloader, mode, None)
    iter_factory = getattr(mode_config, "iter_factory", None)
    return int(getattr(iter_factory, "num_workers", 0))


def _iter_speech_batches(loader, categories):
    """Adapt raw samples to ESPnet2's indexed batches when iteration starts."""
    for index, sample in enumerate(loader):
        language = sample["lid_labels"]
        if not isinstance(language, str) or not language or len(language.split()) != 1:
            raise ValueError(
                "LID statistics require a single language string per sample"
            )
        categories[language].append(str(index))
        yield [str(index)], {"speech": sample["speech"].unsqueeze(0)}


def collect_speech_shapes(config) -> None:
    """Collect raw speech shapes with ESPnet2 without constructing a model.

    Preprocessing is disabled and combined-dataset indices are used as IDs.
    Only speech is forwarded; string language labels are used to write category
    mappings with the same combined indices, including across multiple datasets.

    Args:
        config: Training config containing ``dataset``, ``stats_dir``, and
            ``dataloader``. Each split's ``iter_factory.num_workers`` defaults to 0.

    Returns:
        None: Writes ``speech_shape``, ``batch_keys``, and empty ``stats_keys``
        under ``stats_dir/train`` and ``stats_dir/valid`` using ESPnet2's format.
        Also writes ``category2utt`` and ``lang2utt`` for each combined split,
        plus ``dataset2utt``/``utt2dataset`` using dataset positions as names.

    Raises:
        ValueError: If either split is empty or a language label is not a single string.
    """
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.dataset, resolve=True)
    )
    dataset_config.preprocessor = None
    organizer = instantiate(dataset_config)
    iterators = {}
    categories = {mode: defaultdict(list) for mode in ("train", "valid")}

    for mode in ("train", "valid"):
        dataset = getattr(organizer, mode)
        if len(dataset) == 0:
            raise ValueError(f"Cannot collect speech shapes: {mode} dataset is empty")
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=_get_num_workers(config, mode),
        )
        iterators[mode] = _iter_speech_batches(loader, categories[mode])

    collect_stats(
        model=None,
        train_iter=iterators["train"],
        valid_iter=iterators["valid"],
        output_dir=Path(config.stats_dir),
        ngpu=0,
        log_interval=None,
        write_collected_feats=False,
    )

    for mode, mapping in categories.items():
        lines = "".join(
            f"{language} {' '.join(indices)}\n"
            for language, indices in sorted(mapping.items())
        )
        output_dir = Path(config.stats_dir) / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("category2utt", "lang2utt"):
            (output_dir / name).write_text(lines, encoding="utf-8")

        dataset = getattr(organizer, mode)
        sources = getattr(dataset, "datasets", [dataset])
        offset = 0
        with (
            (output_dir / "dataset2utt").open("w", encoding="utf-8") as dataset2utt,
            (output_dir / "utt2dataset").open("w", encoding="utf-8") as utt2dataset,
        ):
            for source_index, source in enumerate(sources):
                indices = range(offset, offset + len(source))
                if len(source):
                    dataset2utt.write(f"{source_index} {' '.join(map(str, indices))}\n")
                    for index in indices:
                        utt2dataset.write(f"{index} {source_index}\n")
                offset += len(source)
