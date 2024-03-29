import os
import shutil
from pathlib import Path
from typing import Union

import sentencepiece as spm


def prepare_sentences(
    dump_text_paths: Union[str, Path],
    output_path: Union[str, Path],
    remove_characters: str = "",
):
    """
    Create train.txt file for sentencepiece training from the given dump file.

    Args:
        dump_text_paths (Union[str, Path]): Dump text file path.
        output_path (Union[str, Path]): Output directory for train.txt file.
        remove_characters (str): Characters to be removed from the text.
    """
    # Please join the dump set before running this function.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    lines = []
    for dump_text_path in dump_text_paths:
        with open(dump_text_path, "r") as f:
            lines += f.readlines()

    # normalize text
    # remove unrequired characters
    table = str.maketrans("", "", remove_characters)
    lines = [line.translate(table) for line in lines]
    texts = "\n".join(
        [line.split(" ", maxsplit=1)[1].replace("\n", "") for line in lines]
    )

    with open(os.path.join(output_path, "train.txt"), "w") as f:
        f.write(texts)


def train_sentencepiece(
    dump_text_path: Union[str, Path],
    output_path: Union[str, Path],
    vocab_size: int = 5000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    user_defined_symbols: list = [],
):
    """Main function to train sentencepiece model.

    Args:
        dump_text_path (Union[str, Path]): Path to the train.txt file.
        output_path (Union[str, Path]): Output directory to store
            sentencepiece model and vocaburary list.
        vocab_size (int, optional): Vocaburary size. Defaults to 5000.
        character_coverage (float, optional): Character coverage.
            Defaults to 0.9995.
        model_type (str, optional): Model type of sentencepiece. Defaults to "bpe".
        user_defined_symbols (list, optional): User defined symbols.
    """
    # Please prepare sentences before running this function.
    spm.SentencePieceTrainer.Train(
        input=dump_text_path,
        model_prefix=model_type,
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        user_defined_symbols=user_defined_symbols,
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shutil.move(f"{model_type}.model", output_path)
    shutil.move(f"{model_type}.vocab", output_path)

    # create vocab file
    with open(os.path.join(output_path, f"{model_type}.vocab"), "r") as f:
        lines = f.readlines()

    vocabs = (
        ["<blank>", "<unk>"]
        + [line.split("\t")[0] for line in lines][3:]
        + ["<sos/eos>"]
    )
    with open(os.path.join(output_path, "tokens.txt"), "w") as f:
        f.write("\n".join(vocabs))
