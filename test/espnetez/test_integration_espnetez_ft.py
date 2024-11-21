import argparse
from pathlib import Path

import espnetez as ez
from espnet2.bin.asr_inference import Speech2Text as ASRInference
from espnet2.bin.asr_transducer_inference import Speech2Text as RNNTInference
from espnet2.bin.enh_inference import SeparateSpeech as ENHInference
from espnet2.bin.enh_tse_inference import SeparateSpeech as ENHTSEInference
from espnet2.bin.lm_inference import GenerateText as LMInference
from espnet2.bin.mt_inference import Text2Text as MTInference
from espnet2.bin.s2st_inference import Speech2Speech as S2STInference
from espnet2.bin.s2t_inference import Speech2Text as S2TInference
from espnet2.bin.slu_inference import Speech2Understand as SLUInference
from espnet2.bin.spk_inference import Speech2Embedding as SPKInference
from espnet2.bin.st_inference import Speech2Text as STInference
from espnet2.bin.tts_inference import Text2Speech as TTSInference
from espnet2.bin.uasr_inference import Speech2Text as UASRInference
from espnet2.layers.create_adapter_fn import create_lora_adapter

TASK_CLASSES = {
    "asr": ASRInference,
    "asr_transducer": RNNTInference,
    "lm": LMInference,
    "mt": MTInference,
    "slu": SLUInference,
    "tts": TTSInference,
    "gan_tts": TTSInference,
    "uasr": UASRInference,
    "enh": ENHInference,
    "enh_tse": ENHTSEInference,
    "enh_s2t": ASRInference,
    "st": STInference,
    "s2t": S2TInference,
    "s2st": S2STInference,
    "spk": SPKInference,
}

CONFIG_NAMES = {
    "asr": "asr_train_args",
    "asr_transducer": "asr_train_args",
    "lm": "lm_train_args",
    "mt": "mt_train_args",
    "slu": "asr_train_args",
    "tts": "train_args",
    "gan_tts": "train_args",
    "uasr": "uasr_train_args",
    "enh": "enh_train_args",
    "enh_tse": "enh_train_args",
    "enh_s2t": "asr_train_args",
    "st": "st_train_args",
    "s2t": "s2t_train_args",
    "s2st": "train_args",
    "spk": "spk_train_args",
}

LORA_TARGET = [
    "w_1",
    "w_2",
    "merge_proj",  # for tfm and ebf
    "l_last",
    "linear",  # for enh
]


def get_pretrained_model(args):
    exp_dir = args.exp_path / args.task
    if args.task in ("tts", "enh", "enh_tse", "s2st", "spk", "gan_tts"):
        return TASK_CLASSES[args.task](
            exp_dir / "config.yaml",  # config.yaml
            exp_dir / "1epoch.pth",  # checkpoint
        )
    elif args.task == "enh_s2t":
        return ASRInference(
            exp_dir / "config.yaml",
            exp_dir / "1epoch.pth",
            token_type="bpe",
            bpemodel=str(args.data_path / "spm/bpemodel/bpe.model"),
            enh_s2t_task=True,
        )
    else:
        return TASK_CLASSES[args.task](
            exp_dir / "config.yaml",  # config.yaml
            exp_dir / "1epoch.pth",  # checkpoint
            token_type="bpe",
            bpemodel=str(args.data_path / "spm/bpemodel/bpe.model"),
        )


def build_model_fn(args):
    pretrained_model = get_pretrained_model(args)
    if args.task in ("asr", "asr_transducer", "enh_s2t"):
        model = pretrained_model.asr_model
    elif args.task == "lm":
        model = pretrained_model.lm
    elif args.task == "mt":
        model = pretrained_model.mt_model
    elif args.task == "slu":
        model = pretrained_model.asr_model
    elif args.task == "tts":
        model = pretrained_model.tts
    elif args.task == "uasr":
        model = pretrained_model.uasr_model
    elif args.task in ("enh", "enh_tse"):
        model = pretrained_model.enh_model
    elif args.task == "st":
        model = pretrained_model.st_model
    elif args.task == "s2t":
        model = pretrained_model.s2t_model
    elif args.task == "s2st":
        model = pretrained_model.model
    elif args.task == "spk":
        model = pretrained_model.spk_model

    model.train()
    # apply lora
    create_lora_adapter(model, target_modules=LORA_TARGET)
    return model


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
        "--run_finetune",
        action="store_true",
        help="Flag to test finetuning",
    )
    parser.add_argument(
        "--use_discrete_unit",
        action="store_true",
        help="Flag to use discrete unit. Only used for s2st task",
    )
    args = parser.parse_args()

    # In this test we assume that we use mini_an4 dataset.
    data_info = {
        "speech": ["wav.scp", "sound"],
        "text": ["text", "text"],
    }

    # Prepare configurations
    exp_dir = str(args.exp_path / args.task)
    stats_dir = str(args.exp_path / "stats")

    # Get the pretrained model
    pretrained_model = get_pretrained_model(args)
    pretrain_config = getattr(pretrained_model, CONFIG_NAMES[args.task])

    converter = getattr(pretrained_model, "converter", None)
    tokenizer = getattr(pretrained_model, "tokenizer", None)

    finetune_config = ez.config.update_finetune_config(
        args.task, vars(pretrain_config), "../asr1/conf/finetune_with_lora.yaml"
    )

    finetune_config["max_epoch"] = 2
    finetune_config["ngpu"] = 0
    finetune_config["bpemodel"] = str(args.data_path / "spm/bpemodel/bpe.model")

    if args.task == "lm":
        data_info.pop("speech")

    elif args.task == "tts":
        training_config = ez.config.from_yaml(args.task, args.config_path)
        finetune_config["normalize"] = training_config["normalize"]
        finetune_config["pitch_normalize"] = training_config["pitch_normalize"]
        finetune_config["energy_normalize"] = training_config["energy_normalize"]

    elif args.task == "enh":
        data_info = {
            f"speech_ref{i+1}": [f"spk{i+1}.scp", "sound"]
            for i in range(finetune_config["separator_conf"]["num_spk"])
        }
        data_info["speech_mix"] = ["wav.scp", "sound"]

    elif args.task == "enh_tse":
        data_info = {
            "enroll_ref1": ["enroll_spk1.scp", "text"],
            "speech_ref1": ["spk1.scp", "sound"],
        }
        data_info["speech_mix"] = ["wav.scp", "sound"]

    elif args.task == "enh_s2t":
        data_info = {
            "text_spk1": ["text_spk1", "text"],
            "speech_ref1": ["spk1.scp", "sound"],
            "speech": ["wav.scp", "sound"],
        }

    elif args.task == "st":
        data_info["text"] = ["text.lc.rm.en", "text"]
        data_info["src_text"] = ["text", "text"]
        src_file = args.data_path / "en_en_token_list/src_bpe_unigram30"
        finetune_config["bpemodel"] = str(src_file / "bpe.model")

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

    elif args.task == "spk":
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

    trainer = ez.Trainer(
        task=args.task,
        train_config=finetune_config,
        train_dump_dir=args.train_dump_path,
        valid_dump_dir=args.valid_dump_path,
        data_info=data_info,
        output_dir=exp_dir,
        stats_dir=stats_dir,
        ngpu=0,
    )

    if args.run_finetune:
        trainer.train()
