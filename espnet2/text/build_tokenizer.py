from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer
from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.hugging_face_tokenizer import HuggingFaceTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


@typechecked
def build_tokenizer(
    token_type: str,
    bpemodel: Optional[Union[Path, str, Iterable[str]]] = None,
    non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: Optional[str] = None,
    g2p_type: Optional[str] = None,
    nonsplit_symbol: Optional[Iterable[str]] = None,
    # tokenization encode (text2token) args, e.g. BPE dropout, only applied in training
    encode_kwargs: Optional[Dict] = None,
    # only use for whisper
    whisper_language: Optional[str] = None,
    whisper_task: Optional[str] = None,
    sot_asr: bool = False,
) -> AbsTokenizer:
    """
    A helper function to instantiate a tokenizer based on the specified type.

    This function creates an instance of a tokenizer based on the `token_type`
    provided. The function supports various tokenization methods, including BPE,
    Hugging Face, word, character, phoneme, and Whisper tokenizers.

    Attributes:
        token_type (str): The type of tokenizer to instantiate. Must be one of:
            'bpe', 'hugging_face', 'word', 'char', 'phn', or a whisper variant.
        bpemodel (Optional[Union[Path, str, Iterable[str]]]): The path to the BPE
            model file or a string/iterable for models requiring this parameter.
        non_linguistic_symbols (Optional[Union[Path, str, Iterable[str]]]): Symbols
            to be considered non-linguistic. Applicable for word and char tokenizers.
        remove_non_linguistic_symbols (bool): If True, removes non-linguistic symbols
            from the tokenization process. Not implemented for BPE and Hugging Face.
        space_symbol (str): The symbol used to represent spaces in the tokenization.
        delimiter (Optional[str]): The delimiter used for tokenizing text.
        g2p_type (Optional[str]): Type of grapheme-to-phoneme conversion to use.
        nonsplit_symbol (Optional[Iterable[str]]): Symbols that should not be split.
        encode_kwargs (Optional[Dict]): Additional arguments for encoding (text to token).
        whisper_language (Optional[str]): Language to be used for Whisper tokenizer.
        whisper_task (Optional[str]): Task type for Whisper tokenizer (e.g., 'transcribe').
        sot_asr (bool): Whether to use start-of-transcript for ASR.

    Args:
        token_type (str): Type of the tokenizer to build.
        bpemodel (Optional[Union[Path, str, Iterable[str]]]): BPE model for BPE and
            Hugging Face tokenizers.
        non_linguistic_symbols (Optional[Union[Path, str, Iterable[str]]]): Non-linguistic
            symbols for word and char tokenizers.
        remove_non_linguistic_symbols (bool): Flag to remove non-linguistic symbols.
        space_symbol (str): Symbol for spaces.
        delimiter (Optional[str]): Delimiter for word tokenization.
        g2p_type (Optional[str]): G2P conversion type for phoneme tokenization.
        nonsplit_symbol (Optional[Iterable[str]]): Symbols not to be split.
        encode_kwargs (Optional[Dict]): Encoding arguments.
        whisper_language (Optional[str]): Language for Whisper tokenization.
        whisper_task (Optional[str]): Task type for Whisper.
        sot_asr (bool): Start-of-transcript for ASR.

    Returns:
        AbsTokenizer: An instance of a tokenizer based on the specified type.

    Raises:
        ValueError: If `token_type` is not recognized or if `bpemodel` is required
            but not provided for BPE or Hugging Face tokenizers.
        RuntimeError: If `remove_non_linguistic_symbols` is used with BPE or Hugging
            Face tokenizers.

    Examples:
        >>> tokenizer = build_tokenizer("bpe", bpemodel="path/to/bpe.model")
        >>> tokens = tokenizer.encode("Hello world!")
        >>> print(tokens)

        >>> tokenizer = build_tokenizer("word", non_linguistic_symbols=["@"])
        >>> tokens = tokenizer.encode("Hello @ world!")
        >>> print(tokens)

        >>> tokenizer = build_tokenizer("whisper", bpemodel="path/to/model",
        ... whisper_language="en", whisper_task="transcribe")
        >>> tokens = tokenizer.encode("Hello world!")
        >>> print(tokens)
    """
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpe"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not implemented for token_type=bpe"
            )
        if encode_kwargs is None:
            encode_kwargs = dict()
        return SentencepiecesTokenizer(bpemodel, encode_kwargs)

    if token_type == "hugging_face":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "hugging_face"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not "
                + "implemented for token_type=hugging_face"
            )
        return HuggingFaceTokenizer(bpemodel)

    elif token_type == "word":
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            return WordTokenizer(
                delimiter=delimiter,
                non_linguistic_symbols=non_linguistic_symbols,
                remove_non_linguistic_symbols=True,
            )
        else:
            return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
            nonsplit_symbols=nonsplit_symbol,
        )

    elif token_type == "phn":
        return PhonemeTokenizer(
            g2p_type=g2p_type,
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    elif "whisper" in token_type:
        return OpenAIWhisperTokenizer(
            model_type=bpemodel,
            language=whisper_language or "en",
            task=whisper_task or "transcribe",
            added_tokens_txt=non_linguistic_symbols,
            sot=sot_asr,
        )

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, char or phn: " f"{token_type}"
        )
