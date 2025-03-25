
from omegaconf import OmegaConf
from hydra.utils import instantiate

import librosa
import numpy as np

from datasets import load_from_disk, concatenate_datasets
import espnetez as ez
from espnetez.parallel import set_parallel, get_client, get_parallel_config
from espnetez.trainer import ESPnetEZLightningTrainer, LitESPnetModel
from espnetez.data.lhotse_utils import cutset_from_huggingface, HuggingfaceDatasetsBackend
from lhotse.audio.backend import set_current_audio_backend

print("IMPORTED", flush=True)

def load_line(path):
    with open(path, "r") as f:
        data = f.readlines()
    return data


def get_dataset(config):
    # load dataset
    train_dataset = load_from_disk(
        config.dataset.id,
    )["train-clean-100"]
    dev_dataset = concatenate_datasets([
        load_from_disk(
            config.dataset.id,
        )["dev-clean"],
    ])

    # Convert to Cut format
    data_info = {
        "text": lambda x: x["text"],
        "language": lambda x: "English",
        "speaker": lambda x: x['id'].split("-")[0],
        "channel": lambda x: 0,
    }

    # cutset_from_huggingface is parallelized with config.parallel
    train_cuts = cutset_from_huggingface(
        data_info, len(train_dataset), config.dataset.id, "dev-other"
    )
    dev_cuts = cutset_from_huggingface(
        data_info, len(dev_dataset), config.dataset.id, "dev-clean"
    )
    return train_cuts, dev_cuts


def get_dataset_ez(config, tokenize):
    # load dataset
    train_dataset = load_from_disk(
        config.dataset.id,
    )["train-clean-100"]
    dev_dataset = concatenate_datasets([
        load_from_disk(
            config.dataset.id,
        )["dev-clean"],
    ])
    # Convert to Cut format
    data_info = {
        "speech": lambda d: d["audio"]['array'].astype(np.float32),
        "text": lambda d: tokenize(d["text"]).astype(np.int64),
        # "text_prev": lambda d: tokenize("<na>"),
        # "text_ctc": lambda d: tokenize(d["text_ctc"]),
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

    # Set parallelism config
    set_parallel(config.parallel)
    print("Parallelism config set", flush=True)

    # Get datase
    tokenizer = instantiate(config.tokenizer)
    converter = instantiate(config.converter)
    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    train_cuts, dev_cuts = get_dataset_ez(config, tokenize)
    print("Dataset loaded", flush=True)

    set_current_audio_backend(HuggingfaceDatasetsBackend(
        config.dataset.id,
    ))
    print("Set audio backend", flush=True)

    # Define model
    model = LitESPnetModel(
        config.training,
        instantiate(config.model),
        train_dataset=train_cuts,
        valid_dataset=dev_cuts,
    )
    print("Model defined", flush=True)

    lightning_config = {} if not hasattr(config, "lightning") else config.lightning

    trainer = ESPnetEZLightningTrainer(
        config.trainer,
        model=model,
        **lightning_config,
    )
    print("Trainer defined", flush=True)

    trainer.train()

