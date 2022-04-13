""" Useful functions for extracting stresses
"""
import json
import os.path
import pkg_resources
import re
import sys


RUSSIAN_VOWELS = "аеёиоуыэюя"


def load_package_dicts(dictnames_regex = '.?'):
    """ Loads word stresses from packages resources.

    1. dictnames_regex - regex to match loaded dictionaries
    2. return united dictionary
    """
    dict_filenames_re = re.compile(dictnames_regex)
    dict_filenames = pkg_resources.resource_listdir(__package__, 'dicts')
    dictionary = {}
    for dict_filename in dict_filenames:
        if dict_filenames_re.search(os.path.splitext(dict_filename)[0]):
            dict_full_filename = os.path.join('dicts', dict_filename)
            stresses = json.load(
                pkg_resources.resource_stream(__package__, dict_full_filename))
            dictionary = dict(dictionary, **stresses)
    return dictionary


def load_stresses_file(filename):
    """ Load stresses dictionary from `filename`
    """
    with open(filename, 'r') as f:
        js = json.load(f)
    unjsonable_struct = {w: set(strs) for w, strs in js.items()}
    return unjsonable_struct


def save_stresses(stresses, output_filename):
    """ Save dictionary `stressed_words` to stdout.

    1. stresses        - stressed dictionary
    2. output_filename - filename to save
    """
    jsonable_struct = {w: list(strs) for w, strs in stresses.items()}
    with open(output_filename, 'w') as f:
        json.dump(jsonable_struct, f,
                  sort_keys=True, indent=4, ensure_ascii=False)


def is_vowel(v):
    """ Check is `v` a vowel

    1. v - letter
    2. return True if `v` is vowel, False otherwise
    """
    return v in RUSSIAN_VOWELS


def vowel_positions(word):
    """ Finds vowels and their positions in word.

    1. word - word to check
    2. return list of pairs (position, vowel)
    """
    return {(pos + 1): letter for pos, letter in enumerate(word) if is_vowel(letter)}

