import os
import argparse
import torch
import numpy as np
import lightning as L
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from espnet3 import get_espnet_model, save_espnet_config
from espnet3.trainer import ESPnetEZLightningTrainer, LitESPnetModel
from espnet3.parallel import set_parallel
from espnet3.data import DataOrganizer


def load_line(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def main(config_path):
    # Load config
    OmegaConf.register_new_resolver("load_line", load_line)
    config = OmegaConf.load(config_path)

    # Save config
    task = getattr(config, "task", None)
    # save_espnet_config(task, config, config.expdir)

    # Set seed
    if getattr(config, "seed", None) is not None:
        assert isinstance(config.seed, int), "seed should be an integer"
        L.seed_everything(config.seed)

    # Set parallel config
    # set_parallel(config.parallel)

    # Define model
    if task is not None:
        model = get_espnet_model(task, config.model)
    else:
        model = instantiate(config.model)

    model = LitESPnetModel(
        model,
        config,
    )

    # Float32 precision
    torch.set_float32_matmul_precision("high")

    # Setup trainer and run
    trainer = ESPnetEZLightningTrainer(
        model=model,
        expdir=config.expdir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )
    print(trainer)
    print(model)
    # trainer.collect_stats()

    fit_params = {} if not hasattr(config, "fit") else config.fit
    trainer.fit(**fit_params)
    # trainer.validate(**fit_params)

if __name__ == "__main__":
    main("config_globalMVN.yaml")
