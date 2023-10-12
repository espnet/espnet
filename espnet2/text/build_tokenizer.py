from pathlib import Path
from typing import Dict, Iterable, Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer
from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.hugging_face_tokenizer import HuggingFaceTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
    g2p_type: str = None,
    nonsplit_symbol: Iterable[str] = None,
    # tokenization encode (text2token) args, e.g. BPE dropout, only applied in training
    encode_kwargs: Dict = None,
    # only use for whisper
    whisper_language: str = None,
    whisper_task: str = None,
    sot_asr: bool = False,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer"""
    assert check_argument_types()
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
