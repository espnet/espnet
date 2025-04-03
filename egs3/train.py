import os
from argparse import Namespace

import lightning as L
import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from hydra.utils import instantiate
from lhotse.audio.backend import set_current_audio_backend
from omegaconf import OmegaConf

import espnet3 as ez
from espnet3.data import HuggingfaceDatasetsBackend, cutset_from_huggingface
from espnet3.parallel import set_parallel
from espnet3.trainer import ESPnetEZLightningTrainer, LitESPnetModel


def load_line(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [t.strip() for t in data]


# def get_dataset(config):
#     # load dataset
#     train_dataset = load_from_disk(
#         config.dataset.id,
#     )["dev-other"]
#     dev_dataset = concatenate_datasets([
#         load_from_disk(
#             config.dataset.id,
#         )["dev-clean"],
#     ])
#     # Convert to Cut format
#     data_info = {
#         "text": lambda x: x["text"],
#         "language": lambda x: "English",
#         "speaker": lambda x: x['id'].split("-")[0],
#         "channel": lambda x: 0,
#     }
#     # cutset_from_huggingface is parallelized with config.parallel
#     train_cuts = cutset_from_huggingface(
#         data_info, len(train_dataset), config.dataset.id, "dev-other"
#     )
#     dev_cuts = cutset_from_huggingface(
#         data_info, len(dev_dataset), config.dataset.id, "dev-clean"
#     )
#     return train_cuts, dev_cuts


def get_dataset_ez(config, tokenize):
    # load dataset
    train_dataset = load_from_disk(
        config.dataset.id,
    )["train-clean-100"]
    dev_dataset = concatenate_datasets(
        [
            load_from_disk(
                config.dataset.id,
            )["dev-clean"],
        ]
    )
    # Convert to Cut format
    data_info = {
        "speech": lambda d: d["audio"]["array"].astype(np.float32),
        "text": lambda d: tokenize(d["text"]).astype(np.int64),
    }
    # Convert into ESPnet-EZ dataset format
    train_dataset = ez.data.ESPnetEZDataset(train_dataset, data_info=data_info)
    valid_dataset = ez.data.ESPnetEZDataset(dev_dataset, data_info=data_info)
    return train_dataset, valid_dataset


if __name__ == "__main__":
    # load config
    print("Loading config", flush=True)
    config = OmegaConf.load("egs3/config.yaml")
    OmegaConf.register_new_resolver("load_line", load_line)
    print("Config loaded", flush=True)

    # save the configuration for inference
    ez.save_espnet_config("asr", config, os.path.join(config.expdir, "config.yaml"))

    # Set random seed if required
    if getattr(config, "seed", None) is not None:
        assert isinstance(config.seed, int), "seed should be an integer"
        L.seed_everything(config.seed)

    # Set parallelism config
    set_parallel(config.parallel)
    print("Parallelism config set", flush=True)

    # Required if using Huggingface + Lhotse
    # set_current_audio_backend(HuggingfaceDatasetsBackend(
    #     config.dataset.id,
    # ))
    # print("Set audio backend", flush=True)

    # Get datase
    tokenizer = instantiate(config.tokenizer)
    converter = instantiate(config.converter)

    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    train_cuts, dev_cuts = get_dataset_ez(config, tokenize)
    print("Dataset loaded", flush=True)

    # Users can use this function for any model from espnet config.
    # Otherwise you can define your own model here.
    espnet_model = ez.get_espnet_model(
        "asr",
        config.model,
    )
    model = LitESPnetModel(
        espnet_model,
        config,
        train_dataset=train_cuts,
        valid_dataset=dev_cuts,
    )
    print("Model defined", flush=True)

    # Set additional configurations that might be helpful
    torch.set_float32_matmul_precision("high")

    trainer = ESPnetEZLightningTrainer(
        model=model,
        expdir=config.expdir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )
    print("Trainer defined", flush=True)

    fit_params = {} if not hasattr(config, "fit") else config.fit
    trainer.fit(**fit_params)
