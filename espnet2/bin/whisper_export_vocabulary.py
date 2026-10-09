#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from typeguard import typechecked

from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.text.whisper_tokenizer import LANGUAGES_CODE_MAPPING
from espnet2.utils.types import str2bool

dirname = os.path.dirname(__file__)


@typechecked
def export_vocabulary(
    output: str,
    whisper_model: str,
    whisper_language: Optional[str] = "en",
    whisper_task: str = "transcribe",
    log_level: str = "INFO",
    add_token_file_name: str = "none",
    sot_asr: bool = False,
    speaker_change_symbol: str = "<sc>",
):
    try:
        import whisper.tokenizer
    except Exception as e:
        print("Error: whisper is not properly installed.")
        print(
            "Please install whisper with: cd ${MAIN_ROOT}/tools && "
            "./installers/install_whisper.sh"
        )
        raise e

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if output == "-":
        fout = sys.stdout
    else:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        fout = p.open("w", encoding="utf-8")

    whisper_language = LANGUAGES_CODE_MAPPING.get(whisper_language)
    if whisper_language is None:
        raise ValueError("language unsupported for Whisper model")
    if whisper_task not in ["transcribe", "translate"]:
        raise ValueError(f"task: {whisper_task} unsupported for Whisper model")

    if whisper_model == "whisper_en":
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
    elif whisper_model == "whisper_multilingual":
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, language=whisper_language, task=whisper_task
        )
        # Copy before anything is added: get_tokenizer is lru_cached, so the
        # object above is shared with every other caller in the process, and
        # the padding loop below reads its size.
        tokenizer = copy.deepcopy(tokenizer)
        if add_token_file_name != "none":
            _added_tokens = []
            with open(add_token_file_name) as f:
                lines = f.readlines()
                for line in lines:
                    _added_tokens.append(line.rstrip())
            tokenizer.tokenizer.add_tokens(_added_tokens)
    else:
        raise ValueError("tokenizer unsupported:", whisper_model)

    vocab_size = tokenizer.tokenizer.vocab_size + len(
        tokenizer.tokenizer.get_added_vocab()
    )
    if whisper_model == "whisper_en":
        vocab_size = vocab_size - 1

    for i in range(vocab_size):
        # take care of special char for <space>
        tkn = tokenizer.tokenizer.convert_ids_to_tokens(i).replace("Ġ", " ")
        fout.write(tkn + "\n")

    # NOTE (Shih-Lun): extra tokens (for timestamped ASR) not
    #                  stored in the wrapped tokenizer
    full_vocab_size = 51865 if whisper_model == "whisper_multilingual" else 51864

    for i in range(full_vocab_size - vocab_size):
        fout.write(f"<|{i * 0.02:.2f}|>" + "\n")

    if sot_asr:
        # Append only a symbol the vocabulary does not already have. A symbol
        # that is already one BPE token, such as "????" (id 25629), would
        # otherwise appear twice: token_list.index and the tokenizer both
        # resolve the first, leaving the appended row unreachable and the
        # model one row wider than the alphabet it can emit.
        existing = tokenizer.tokenizer.convert_tokens_to_ids(speaker_change_symbol)
        if existing is None or existing == tokenizer.tokenizer.unk_token_id:
            # An unknown symbol maps to the unk id, which for Whisper is
            # <|endoftext|>; a symbol literally spelled that way is not a
            # sensible separator and is treated as absent.
            fout.write(speaker_change_symbol + "\n")
            logging.info("Appended %s to the vocabulary", speaker_change_symbol)
        else:
            logging.warning(
                "%s is already token id %d; not appending. The exported list "
                "is one row shorter than the same command produced before "
                "this guard, so a checkpoint trained against the older list "
                "will not load against it.",
                speaker_change_symbol,
                existing,
            )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Whisper vocabulary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output", "-o", required=True, help="Output text. - indicates sys.stdout"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        required=True,
        help="Whisper model type",
    )
    parser.add_argument(
        "--add_token_file_name",
        type=str,
        default="none",
        help="File name for added tokens",
    )
    parser.add_argument(
        "--whisper_language",
        type=str,
        default="en",
        help="Language for Whisper multilingual tokenizer",
    )
    parser.add_argument(
        "--whisper_task",
        type=str,
        default="transcribe",
        help="Task for Whisper multilingual tokenizer",
    )
    parser.add_argument(
        "--sot_asr",
        type=str2bool,
        default=False,
        required=False,
        help="Whether SOT-style training is used in Whisper",
    )
    parser.add_argument(
        "--speaker_change_symbol",
        type=str,
        default="<sc>",
        required=False,
        help="Whether SOT-style training is used in Whisper",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    export_vocabulary(**kwargs)


if __name__ == "__main__":
    main()
