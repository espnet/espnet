#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Расшифровка сокращений в англоязычном тексте в соответствии с таблицей:

Сокращение   |    Расшифровка
-------------------------------
    mrs             misess
    mr              mister
    dr              doctor
    st              saint
    co              company
    jr              junior
    maj             major
    gen             general
    drs             doctors
    rev             reverend
    lt              lieutenant
    hon             honorable
    sgt             sergeant
    capt            captain
    esq             esquire
    ltd             limited
    col             colonel
    ft              fort
'''

import re

# Список из пар (регулярное_выражение, замена) для каждого сокращения
abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations_en(text):
    ''' Расшифровка сокращений в тексте в соответствии с таблицей. '''
    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)
    return text
