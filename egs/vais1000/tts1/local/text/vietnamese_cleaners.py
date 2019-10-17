# -*- coding: utf-8 -*-

""" Created on 10:19 AM, 10/17/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
    Some of this code is taken from `espnet` cleaner module.
"""

""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
1. "english_cleaners" for English text
2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
the Unidecode library (https://pypi.python.org/pypi/Unidecode)
3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
the symbols in symbols.py to match your data).
'''

from regex_tokenize import tokenize
import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def vietnamese_naive_cleaner(text):
    ''' (Naive) Pipeline for vietnamese text. This does not include transcribing number and abbreviation'''

    return tokenize(text, format='text')
