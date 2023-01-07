#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args

def export_vocabulary(
    output: str,
    whisper_model: str,
    log_level: str
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

    assert check_argument_types()

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

    if whisper_model == "whisper_en":
        tokenizer = whisper.tokenizer.get_tokenizer(
                                                multilingual=False
                                            )
    # TODO (Shih-Lun): should support feeding in 
    #                  different languages (default is en)
    elif whisper_model == "whisper_multilingual":
        tokenizer = whisper.tokenizer.get_tokenizer(
                                                multilingual=True,
                                                language=None
                                            )
    else:
        raise ValueError("tokenizer unsupported:", whisper_model)

    vocab_size = tokenizer.tokenizer.vocab_size + \
                 len(tokenizer.tokenizer.get_added_vocab())

    for i in range(vocab_size):
        # take care of special char for <space>
        tkn = tokenizer.tokenizer.convert_ids_to_tokens(i).replace("Ġ", " ")
        fout.write(tkn + '\n')
    
    # NOTE (Shih-Lun): extra tokens (for timestamped ASR) not
    #                  stored in the wrapped tokenizer
    full_vocab_size = 51865 if whisper_model == "whisper_multilingual" \
                            else 51864
    for i in range(full_vocab_size - vocab_size):
        fout.write("()" + '\n')



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

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    export_vocabulary(**kwargs)


if __name__ == "__main__":
    main()