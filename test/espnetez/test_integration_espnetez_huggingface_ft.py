import argparse
from itertools import islice
from pathlib import Path

import numpy as np

try:
    from datasets import Audio, load_dataset
except ImportError:
    raise ImportError(
        "datasets is not installed. Please install it using 'pip install datasets'."
    )

from espnet_model_zoo.downloader import ModelDownloader

import espnetez as ez

TASK_CLASSES = ["svs", "gan_svs"]


def get_data_info(task, dataset_name, model_name):
    if task in ["svs", "gan_svs"] and dataset_name.startswith("espnet"):
        data_info = {
            "singing": lambda d: d["audio"]["array"].astype(np.float32),
            "score": lambda d: (
                d["tempo"],
                list(
                    zip(
                        d["note_start_times"],
                        d["note_end_times"],
                        d["note_lyrics"],
                        d["note_midi"],
                        d["note_phns"],
                    )
                ),
            ),
            "text": lambda d: d["transcription"],
            "label": lambda d: (
                np.array(list(zip(d["phn_start_times"], d["phn_end_times"]))),
                d["phns"],
            ),
        }
        if (
            dataset_name == "espnet/ace-kising-segments"
            and model_name == "espnet/aceopencpop_svs_visinger2_40singer_pretrain"
        ):
            # Speaker-to-ID mapping
            singer2sid = {
                "barber": 3,
                "blanca": 30,
                "changge": 5,
                "chuci": 19,
                "chuming": 4,
                "crimson": 1,
                "david": 28,
                "ghost": 27,
                "growl": 25,
                "hiragi-yuki": 22,
                "huolian": 13,
                "kuro": 2,
                "lien": 29,
                "liyuan": 9,
                "luanming": 21,
                "luotianyi": 31,
                "namine": 8,
                "orange": 12,
                "original": 32,
                "qifu": 16,
                "qili": 15,
                "qixuan": 7,
                "quehe": 6,
                "ranhuhu": 11,
                "steel": 26,
                "tangerine": 23,
                "tarara": 20,
                "tuyuan": 24,
                "wenli": 10,
                "xiaomo": 17,
                "xiaoye": 14,
                "yanhe": 33,
                "yuezhengling": 34,
                "yunhao": 18,
            }
            data_info["sids"] = lambda d: (
                np.array([singer2sid[d["singer"]]]) if "singer" in d else None
            )
        return data_info
    else:
        raise NotImplementedError(
            f"Data info not implemented for task '{task}' and dataset '{dataset_name}'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=TASK_CLASSES)
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name from ESPnet Model Zoo"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name from HuggingFace Dataset Hub",
    )
    parser.add_argument(
        "--use_ez_preprocessor",
        action="store_true",
        default=False,
        help="Use ESPnetEZ preprocessor",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Directory for outputs"
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=None, help="Directory for cached files"
    )
    parser.add_argument(
        "--run_finetune",
        action="store_true",
        default=False,
        help="Flag to run fine-tuning",
    )
    args = parser.parse_args()

    # Download model from HuggingFace Hub
    downloader = ModelDownloader(args.cache_dir)
    pretrain = downloader.download_and_unpack(args.model_name)
    pretrain_config = ez.config.from_yaml(args.task, pretrain["train_config"])
    pretrain_config["model_file"] = pretrain["model_file"]

    # Load dataset and resample audio
    streaming_dataset = load_dataset(
        args.dataset_name,
        split="train",
        streaming=True,
    ).cast_column("audio", Audio(sampling_rate=pretrain_config["fs"]))
    subset = list(islice(streaming_dataset, 100))
    train_dataset = subset[:90]
    valid_dataset = subset[90:100]

    # Define data mapping for the dataset
    data_info = get_data_info(args.task, args.dataset_name, args.model_name)

    # Wrap datasets using ESPnetEZDataset
    train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
    valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)

    # Customize fine-tuning config
    finetune_config = pretrain_config.copy()
    finetune_config.update(
        {
            "batch_size": 1,
            "num_workers": 1,
            "max_epoch": 2,
            "num_iters_per_epoch": None,
            "use_ez_preprocessor": args.use_ez_preprocessor,
        }
    )

    # Define output directories
    EXP_DIR = args.output_dir / "exp"
    STATS_DIR = args.output_dir / "stats"

    # Initialize trainer
    trainer = ez.Trainer(
        task=args.task,
        train_config=finetune_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        data_info=data_info,
        output_dir=str(EXP_DIR),
        stats_dir=str(STATS_DIR),
        ngpu=0,
    )

    # Collect stats
    trainer.train_config.normalize = None
    trainer.train_config.pitch_normalize = None
    trainer.train_config.energy_normalize = None
    trainer.collect_stats()

    trainer.train_config.normalize = finetune_config["normalize"]
    trainer.train_config.pitch_normalize = finetune_config["pitch_normalize"]
    trainer.train_config.normalize_conf["stats_file"] = (
        f"{STATS_DIR}/train/feats_stats.npz"
    )
    trainer.train_config.pitch_normalize_conf["stats_file"] = (
        f"{STATS_DIR}/train/pitch_stats.npz"
    )

    # Fine-tune the model
    if args.run_finetune:
        trainer.train()
