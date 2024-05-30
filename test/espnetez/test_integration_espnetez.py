import argparse
from pathlib import Path

import espnetez as ez

TASK_CLASSES = [
    "asr",
    "gan_tts",
    # "hubert",
    "lm",
    # "s2t",
    "slu",
    # "st",
    "tts",
    "uasr",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASK_CLASSES,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="path to data",
    )
    parser.add_argument(
        "--train_dump_path",
        type=Path,
        required=True,
        help="path to dump",
    )
    parser.add_argument(
        "--valid_dump_path",
        type=Path,
        required=True,
        help="path to valid dump",
    )
    parser.add_argument(
        "--exp_path",
        type=Path,
        required=True,
        help="path to exp",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="path to config yaml file",
    )
    parser.add_argument(
        "--train_sentencepiece_model",
        action="store_true",
        help="Flag to train sentencepiece model",
    )
    parser.add_argument(
        "--run_collect_stats",
        action="store_true",
        help="Flag to run collect stats",
    )
    parser.add_argument(
        "--run_train",
        action="store_true",
        help="Flag to test training",
    )
    args = parser.parse_args()

    # In this test we assume that we use mini_an4 dataset.
    data_info = {
        "speech": ["wav.scp", "sound"],
        "text": ["text", "text"],
    }

    # Train sentencepiece model
    if args.task == "lm":
        user_defined_symbols = ["<generatetext>"]
        data_info.pop("speech")
    else:
        user_defined_symbols = []

    # Tokenize if tts
    if args.task == "tts" or args.task == "gan_tts":
        ez.preprocess.prepare_sentences(
            [args.train_dump_path / "text"], args.data_path / "tokenize"
        )
        ez.preprocess.tokenize(
            input=str(args.data_path / "tokenize" / "train.txt"),
            output=str(args.train_dump_path / "tokenize" / "tokenized.txt"),
            token_type="phn",
            cleaner="tacotron",
            g2p="g2p_en",
        )
        if args.train_sentencepiece_model:
            ez.preprocess.train_sentencepiece(
                args.train_dump_path / "tokenize" / "tokenized.txt",
                args.data_path / "spm/bpemodel",
                vocab_size=50,
                character_coverage=0.9995,
                user_defined_symbols=user_defined_symbols,
            )

    elif args.train_sentencepiece_model:
        ez.preprocess.prepare_sentences(
            [args.train_dump_path / "text"], args.data_path / "spm"
        )
        ez.preprocess.train_sentencepiece(
            args.data_path / "spm/train.txt",
            args.data_path / "spm/bpemodel",
            vocab_size=50,
            character_coverage=0.9995,
            user_defined_symbols=user_defined_symbols,
        )

    # Prepare configurations
    exp_dir = str(args.exp_path / args.task)
    stats_dir = str(args.exp_path / "stats")

    training_config = ez.config.from_yaml(args.task, args.config_path)
    training_config["max_epoch"] = 1
    training_config["ngpu"] = 0
    training_config["bpemodel"] = str(args.data_path / "spm/bpemodel/bpe.model")
    with open(args.data_path / "spm/bpemodel/tokens.txt", "r") as f:
        tokens = [t.replace("\n", "") for t in f.readlines()]
        training_config["token_list"] = tokens

    if args.task == "tts" or args.task == "gan_tts":
        normalize = training_config["normalize"]
        training_config["normalize"] = None
        pitch_normalize = training_config["pitch_normalize"]
        training_config["pitch_normalize"] = None
        energy_normalize = training_config["energy_normalize"]
        training_config["energy_normalize"] = None

    trainer = ez.Trainer(
        task=args.task,
        train_config=training_config,
        train_dump_dir=args.train_dump_path,
        valid_dump_dir=args.valid_dump_path,
        data_info=data_info,
        output_dir=exp_dir,
        stats_dir=stats_dir,
        ngpu=0,
    )
    if args.run_collect_stats:
        trainer.collect_stats()

    if args.task == "tts":
        training_config["normalize"] = normalize
        training_config["pitch_normalize"] = pitch_normalize
        training_config["energy_normalize"] = energy_normalize
        trainer = ez.Trainer(
            task=args.task,
            train_config=training_config,
            train_dump_dir=args.train_dump_path,
            valid_dump_dir=args.valid_dump_path,
            data_info=data_info,
            output_dir=exp_dir,
            stats_dir=stats_dir,
            ngpu=0,
        )

    if args.run_train:
        trainer.train()
