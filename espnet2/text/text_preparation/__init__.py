#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для очистки, нормализации и перевода текста в последовательность чисел и обратно.
'''

from text_preparation.text_normalization import normalize_text, normalize_text_len, LANGUAGES, ctc_symbol_to_id
from text_preparation.text_and_sequences import text_to_sequence, sequence_to_text, get_symbols_length, get_ctc_symbols_length
