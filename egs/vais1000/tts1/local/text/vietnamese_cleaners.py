# -*- coding: utf-8 -*-

"""Vietnamese cleaner.

Created on 10:19 AM, 10/17/19
@author: ngunhuconchocon
@brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
        Vietnamese cleaner. This is a naive implementationm which only seperate punctuation
        and handle some abbreviation. You should see `regex_tokenize.py` for more details.
        This must be updated later for a "cleaner" cleaner
"""

from regex_tokenize import tokenize
from vietnameseNormUniStd import UniStd


def vietnamese_cleaner(text):
    """Perform Vietnamese cleaning.

    Handle the Vietnamese oldstyle of putting tones (òa or oà, úy or uý, ...).
    This action can directly benefit the result if you train the model with letter.
    In case of phoneme training, this cleaner will facilitate the dictionary
    (syllable->phonemes) preparation process.

    Many thanks to Thang Tat Vu and Thanh-Le Ha.

    """
    text = UniStd(text)
    return tokenize(text, format='text')
