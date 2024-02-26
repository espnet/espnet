import re

from tacotron_cleaner.cleaners import (
    collapse_whitespace,
    convert_to_ascii,
    expand_abbreviations,
    expand_numbers,
    expand_symbols,
    lowercase,
)


def remove_extra_chars(text):
    return re.sub(r"[^a-z.,?!'\s]", "", text)


def remove_quotes(text):
    text = re.sub(
        r"([a-z])'([a-z])", r'\1"\2', text
    )  # Save single quote inside words as double quote
    text = text.replace("'", "")  # remove single quotes
    text = text.replace('"', "'")  # put back single quotes inside of words
    return text


def space_after_punc(text):
    text = re.sub(r"([a-z])\.'([a-z])", r"\1'\2", text)
    text = re.sub(r"([a-z][.,?!])(?=[a-z])", r"\1 ", text)
    return text


def mfa_english_cleaner(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = expand_symbols(text)
    text = remove_extra_chars(text)
    text = remove_quotes(text)
    text = collapse_whitespace(text)
    text = space_after_punc(text)
    return text
