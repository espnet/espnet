import argparse

import lightning as L
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from dask.distributed import WorkerPlugin, as_completed, get_worker

from espnet3 import get_espnet_model, save_espnet_config
from espnet3.utils.config import load_config_with_defaults
from espnet3.preprocess import train_sentencepiece
from espnet3.trainer import ESPnetEZLightningTrainer, LitESPnetModel
from espnet3.parallel import get_client, set_parallel


class TokenizerPlugin(WorkerPlugin):
    def __init__(self, dataset_config):
        # remove valid/test config
        dataset_config.pop("test")
        self.dataset_config = dataset_config
    
    def setup(self, worker):
        worker.dataset = instantiate(self.dataset_config)


def train_tokenizer(config):
    config.dataset.preprocessor = None
    organizer = instantiate(config.dataset)
    train_dataset_length = len(organizer.train)
    del organizer

    def get_text(idx: int):
        worker = get_worker()
        dataset = worker.dataset
        return dataset.train.get_text(idx)
    
    plugin = TokenizerPlugin(config.dataset)
    set_parallel(config.parallel)

    with get_client(plugin=plugin) as client, \
        open("train_text.txt", "w", encoding="utf-8") as f:
        futures = client.map(get_text, list(range(train_dataset_length)))
        for future in tqdm(as_completed(futures), total=train_dataset_length):
            text = future.result()
            f.write(text + "\n")
            f.flush()

    train_sentencepiece(
        dump_text_path="train_text.txt",
        output_path="sentencepiece_model",
        vocab_size=config.vocab_size,
        character_coverage=0.995,
        model_type="bpe",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    parser.add_argument(
        "--train_tokenizer", action="store_true", help="Train tokenizer before training"
    )
    parser.add_argument(
        "--collect_stats", action="store_true", help="Run collect_stats before training"
    )
    args = parser.parse_args()

    # Load config
    config = load_config_with_defaults(args.config)

    if args.train_tokenizer:
        print("==> Training tokenizer...")
        train_tokenizer(config)

    # Set seed
    if getattr(config, "seed", None) is not None:
        assert isinstance(config.seed, int), "seed should be an integer"
        L.seed_everything(config.seed)

    # Prepare for collect_stats
    normalize = None
    normalize_conf = None
    if args.collect_stats:
        if "normalize" in config.model:
            normalize = config.model.pop("normalize")
        if "normalize_conf" in config.model:
            normalize_conf = config.model.pop("normalize_conf")

    task = getattr(config, "task", None)
    model = get_espnet_model(task, config.model) if task else instantiate(config.model)
    lit_model = LitESPnetModel(model, config)

    # Float32 precision
    torch.set_float32_matmul_precision("high")

    # Setup trainer
    trainer = ESPnetEZLightningTrainer(
        model=lit_model,
        expdir=config.expdir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )

    if args.collect_stats:
        print("==> Running collect_stats...", flush=True)
        trainer.collect_stats()

        if normalize is not None:
            config.model["normalize"] = normalize
        if normalize_conf is not None:
            config.model["normalize_conf"] = normalize_conf

        model = (
            get_espnet_model(task, config.model) if task else instantiate(config.model)
        )
        lit_model = LitESPnetModel(model, config)

        trainer = ESPnetEZLightningTrainer(
            model=lit_model,
            expdir=config.expdir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

    # save espnet-like config for inference
    if task:
        save_espnet_config(task, config, config.expdir)

    fit_params = {} if not hasattr(config, "fit") else config.fit
    trainer.fit(**fit_params)


if __name__ == "__main__":
    main()
