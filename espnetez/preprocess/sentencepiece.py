import os
import shutil
from pathlib import Path
from typing import Union

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model
from sentencepiece.sentencepiece_model_pb2 import ModelProto
from torch import nn

from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.token_id_converter import TokenIDConverter


def prepare_sentences(
    dump_text_paths: Union[str, Path],
    output_path: Union[str, Path],
    remove_characters: str = "",
):
    """
    Create a training text file for SentencePiece model training from the
    provided dump text files.

    This function consolidates multiple text files into a single `train.txt`
    file, which is formatted for use in SentencePiece training. It also
    provides an option to remove specified characters from the text before
    writing to the output file.

    Args:
        dump_text_paths (Union[str, Path]):
            A single dump text file path or a list of paths to the dump
            text files that will be processed.
        output_path (Union[str, Path]):
            The directory where the `train.txt` file will be saved.
            If the directory does not exist, it will be created.
        remove_characters (str, optional):
            A string containing characters to be removed from the text.
            Defaults to an empty string, meaning no characters will be
            removed.

    Raises:
        FileNotFoundError: If any of the dump text files do not exist.
        IOError: If there is an error reading from the dump text files
        or writing to the output path.

    Examples:
        >>> prepare_sentences("data/dump.txt", "output", remove_characters=",.!")
        This will create an `output/train.txt` file from `data/dump.txt`,
        removing commas, periods, and exclamation marks from the text.

        >>> prepare_sentences(["data/dump1.txt", "data/dump2.txt"], "output")
        This will create an `output/train.txt` file by concatenating
        `data/dump1.txt` and `data/dump2.txt` without removing any characters.

    Note:
        Ensure that the input dump text files are properly formatted, as
        the function expects each line to have a space-separated format
        where the text to be processed is after the first space.
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
    """
    Main function to train a SentencePiece model.

    This function trains a SentencePiece model using the provided training
    data and saves the resulting model and vocabulary files to the specified
    output directory. The model can be customized through various parameters
    such as vocabulary size, character coverage, and model type.

    Args:
        dump_text_path (Union[str, Path]): Path to the `train.txt` file
            containing the training data for the SentencePiece model.
        output_path (Union[str, Path]): Output directory where the trained
            SentencePiece model and vocabulary list will be stored.
        vocab_size (int, optional): The size of the vocabulary to be generated
            by the SentencePiece model. Defaults to 5000.
        character_coverage (float, optional): The character coverage rate
            for the model, which indicates the percentage of characters in
            the training data that should be covered. Defaults to 0.9995.
        model_type (str, optional): The type of model to be trained.
            Options include 'bpe' (Byte Pair Encoding), 'unigram',
            'char', and 'word'. Defaults to "bpe".
        user_defined_symbols (list, optional): A list of user-defined symbols
            that should be included in the model. Defaults to an empty list.

    Raises:
        FileNotFoundError: If the specified `dump_text_path` does not exist.
        Exception: If the training of the SentencePiece model fails for any
            reason.

    Examples:
        >>> train_sentencepiece(
        ...     dump_text_path='path/to/train.txt',
        ...     output_path='path/to/output',
        ...     vocab_size=8000,
        ...     character_coverage=0.995,
        ...     model_type='unigram',
        ...     user_defined_symbols=['<user_sym1>', '<user_sym2>']
        ... )

    Note:
        Ensure that the `train.txt` file has been prepared using the
        `prepare_sentences` function before calling this function.
        The output directory will be created if it does not already exist.
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


def add_special_tokens(
    tokenizer, converter, embedding, special_tokens, insert_after="<st_zho>"
):
    """Add special tokens to the tokenizer.
    For detailed usage, please refer to the demo notebook for ESPnetEZ with SLU task.

    Args:
        tokenizer: Sentencepiece tokenizer.
        converter: Sentencepiece converter.
        embedding: nn.Embedding object.
        special_tokens (list): List of special tokens.

    Returns:
        Tuple(
            tokenizer: new tokenizer,
            converter: new converter,
            embedding: new embedding,
        )
    """
    token_list = converter.token_list

    # First, check if the special tokens are already in the token list.
    add_token_list = []
    for token in special_tokens:
        if token not in token_list:
            # add token to the token list
            add_token_list.append(token)

    # Then append tokens into the token_list and sentencepiece model.
    insert_position = token_list.index(insert_after) + 1
    new_token_list = (
        token_list[:insert_position] + add_token_list + token_list[insert_position:]
    )
    new_converter = TokenIDConverter(new_token_list, converter.unk_symbol)

    new_embedding = nn.Embedding(
        embedding.num_embeddings + len(add_token_list), embedding.embedding_dim
    )
    new_embedding.weight.data[:insert_position] = embedding.weight[:insert_position]
    new_embedding.weight.data[insert_position + len(add_token_list) :] = (
        embedding.weight[insert_position:]
    )

    m = model.ModelProto()
    m.ParseFromString(open(tokenizer.model, "rb").read())
    for token in add_token_list:
        p = ModelProto.SentencePiece(piece=token, score=0.0)
        m.pieces.insert(insert_position, p)

    new_model_path = (
        Path(tokenizer.model).parent / f"{Path(tokenizer.model).stem} _sp.model"
    )
    with open(new_model_path, "wb") as f:
        f.write(m.SerializeToString())

    new_tokenizer = SentencepiecesTokenizer(
        new_model_path, encode_kwargs=tokenizer.encode_kwargs
    )

    return new_tokenizer, new_converter, new_embedding
