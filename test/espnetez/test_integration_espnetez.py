import argparse
from pathlib import Path

import yaml

import espnetez as ez

TASK_CLASSES = [
    "asr",
    "asr_transducer",
    "gan_tts",
    "enh",
    "enh_tse",
    "enh_s2t",
    "hubert",
    "lm",
    "mt",
    "s2t",
    "s2st",
    "slu",
    "st",
    "tts",
    "uasr",
    "spk",
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
    parser.add_argument(
        "--use_discrete_unit",
        action="store_true",
        help="Flag to use discrete unit. Only used for s2st task",
    )
    parser.add_argument(
        "--variable_num_refs",
        action="store_true",
        help="Flag to use variable_num_refs. Only used for ENH task",
    )
    args = parser.parse_args()

    # In this test we assume that we use mini_an4 dataset.
    data_info = {
        "speech": ["wav.scp", "sound"],
        "text": ["text", "text"],
    }

    # configuration
    training_config = ez.config.from_yaml(args.task, args.config_path)
    training_config["max_epoch"] = 1
    training_config["ngpu"] = 0
    training_config["bpemodel"] = str(args.data_path / "spm/bpemodel/bpe.model")

    # Task specific settings
    if args.task == "lm":
        user_defined_symbols = ["<generatetext>"]
        data_info.pop("speech")
    elif args.task == "s2t":
        user_defined_symbols = ["<sos>", "<eos>", "<sop>", "<na>"]
        # add timestamps
        user_defined_symbols += ["<notimestamps>"]
        user_defined_symbols += [f"<{i*0.02:.2f}>" for i in range(1501)]
    else:
        user_defined_symbols = []

    if args.task == "tts" or args.task == "gan_tts":
        normalize = training_config["normalize"]
        training_config["normalize"] = None
        pitch_normalize = training_config["pitch_normalize"]
        training_config["pitch_normalize"] = None
        energy_normalize = training_config["energy_normalize"]
        training_config["energy_normalize"] = None

    # set data_info for specific tasks
    if args.task == "enh":
        data_info = {
            f"speech_ref{i+1}": [f"spk{i+1}.scp", "sound"]
            for i in range(training_config["separator_conf"]["num_spk"])
        }
        data_info["speech_mix"] = ["wav.scp", "sound"]

    elif args.task == "enh_tse":
        spk_type = "sound" if not args.variable_num_refs else "variable_columns_sound"
        data_info = {}
        data_info["speech_mix"] = ["wav.scp", "sound"]
        data_info["enroll_ref1"] = ["enroll_spk1.scp", "text"]
        data_info["speech_ref1"] = ["spk1.scp", spk_type]
        if (
            "num_spk" in training_config["preprocessor_conf"]
            and training_config["preprocessor_conf"]["num_spk"] > 1
        ):
            data_info["category"] = ["utt2category", "text"]

    elif args.task == "enh_s2t":
        data_info = {
            "text_spk1": ["text_spk1", "text"],
            "speech_ref1": ["spk1.scp", "sound"],
            "speech": ["wav.scp", "sound"],
        }
        training_config["text_name"] = ["text", "text_spk1"]

    elif args.task == "hubert":
        data_info["text"] = ["text.km.kmeans_iter0_mfcc_train_nodev_portion1.0", "text"]
        training_config["num_classes"] = 10

    elif args.task == "st":
        data_info["text"] = ["text.lc.rm.en", "text"]
        data_info["src_text"] = ["text", "text"]

    elif args.task == "mt":
        data_info.pop("speech")
        data_info["src_text"] = ["text.ts.mfcc_km10", "text"]
        data_info["text"] = ["text.ts.en", "text"]

    elif args.task == "s2t":
        data_info["speech"] = ["wav.scp", "kaldi_ark"]
        data_info["text_prev"] = ["text.prev", "text"]
        data_info["text_ctc"] = ["text.ctc", "text"]

    elif args.task == "s2st":
        data_info = {
            "src_speech": ["wav.scp.en", "kaldi_ark"],
            "tgt_speech": ["wav.scp.es", "kaldi_ark"],
            "tgt_text": ["text.es", "text"],
            "src_text": ["text.en", "text"],
        }
        if args.use_discrete_unit:
            discrete_file = "text.km.hubert_layer6_5.es.unique"
            data_info["tgt_speech"] = [discrete_file, "text"]
            discrete_folder = "en_es_token_list/discrete_unit.hubert_layer6_5"
            token_text = args.data_path / discrete_folder / "tokens.txt"
            training_config["unit_token_list"] = str(token_text)

    elif args.task == "spk":
        data_info = {
            "train": {
                "speech": ["wav.scp", "sound"],
            },
            "valid": {
                "speech": ["trial.scp", "sound"],
            },
        }
        training_config["spk2utt"] = str(args.train_dump_path / "spk2utt")
        training_config["spk_num"] = 9
        training_config["use_preprocessor"] = False

        # load spk config
        with open("conf/train_rawnet3_debug.yaml", "r") as f:
            spk_config = yaml.load(f, Loader=yaml.FullLoader)
        training_config.update(spk_config)

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
        if args.task == "mt":
            ez.preprocess.prepare_sentences(
                [args.train_dump_path / "text.ts.en"], args.data_path / "spm"
            )
        else:
            ez.preprocess.prepare_sentences(
                [args.train_dump_path / "text"], args.data_path / "spm"
            )

        if args.task == "s2t":
            vocab_size = 50 + 5 + 1501
        else:
            vocab_size = 50

        ez.preprocess.train_sentencepiece(
            args.data_path / "spm/train.txt",
            args.data_path / "spm/bpemodel",
            vocab_size=vocab_size,
            character_coverage=0.9995,
            user_defined_symbols=user_defined_symbols,
        )

    # token related configurations
    if args.task == "hubert":
        training_config["bpemodel"] = None
        training_config["token_type"] = "word"
        token_folder = "noinfo_token_list_kmeans_iter0_mfcc_10clusters"
        with open(args.data_path / token_folder / "word/tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["token_list"] = tokens
    elif args.task == "st":
        bpe_file = args.data_path / "en_en_token_list/tgt_bpe_unigram30"
        src_file = args.data_path / "en_en_token_list/src_bpe_unigram30"
        training_config["src_token_type"] = "bpe"
        training_config["bpemodel"] = str(bpe_file / "bpe.model")
        training_config["src_bpemodel"] = str(src_file / "bpe.model")

        with open(bpe_file / "tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["token_list"] = tokens
        with open(src_file / "tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["src_token_list"] = tokens
    elif args.task == "mt":
        bpe_file = args.data_path / "token_list/char"
        src_file = args.data_path / "token_list/char_mfcc_km10"
        training_config["token_type"] = "char"
        training_config["src_token_type"] = "char"
        training_config["bpemodel"] = None
        training_config["src_bpemodel"] = None

        with open(bpe_file / "tgt_tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["token_list"] = tokens
        with open(src_file / "src_tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["src_token_list"] = tokens
    elif args.task == "s2st":
        token_folder = args.data_path / "en_es_token_list/char"
        training_config["bpemodel"] = None
        training_config["src_bpemodel"] = None
        training_config["token_type"] = "char"
        training_config["tgt_token_type"] = "char"
        training_config["src_token_type"] = "char"

        with open(token_folder / "tgt_tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["tgt_token_list"] = tokens
            training_config["token_list"] = tokens
        with open(token_folder / "src_tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["src_token_list"] = tokens

    elif (args.data_path / "spm/bpemodel/tokens.txt").is_file():
        with open(args.data_path / "spm/bpemodel/tokens.txt", "r") as f:
            tokens = [t.replace("\n", "") for t in f.readlines()]
            training_config["token_list"] = tokens
    else:
        training_config["token_list"] = []

    # Prepare configurations
    exp_dir = str(args.exp_path / args.task)
    stats_dir = str(args.exp_path / "stats")

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

    update_trainer = False
    if args.task == "tts":
        update_trainer = True
        training_config["normalize"] = normalize
        training_config["pitch_normalize"] = pitch_normalize
        training_config["energy_normalize"] = energy_normalize
    elif args.task == "spk":
        update_trainer = True
        data_info = {
            "train": {
                "speech": ["wav.scp", "sound"],
                "spk_labels": ["utt2spk", "text"],
            },
            "valid": {
                "speech": ["trial.scp", "sound"],
                "speech2": ["trial2.scp", "sound"],
                "spk_labels": ["trial_label", "text"],
            },
        }
        training_config = ez.config.from_yaml(args.task, args.config_path)
        training_config["ngpu"] = 0
        training_config["bpemodel"] = str(args.data_path / "spm/bpemodel/bpe.model")
        training_config["token_list"] = []
        training_config["spk2utt"] = str(args.train_dump_path / "spk2utt")
        training_config["spk_num"] = 9  # 3 spk * (0.9, 1.0, 1.1)

    elif args.task == "enh_s2t" and args.variable_num_refs:
        update_trainer = True
        data_info["category"] = ["utt2category", "text"]

    if update_trainer:
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
