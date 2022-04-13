#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Определяет набор допустимых символов, поддерживаемых моделью. Основано на https://github.com/keithito/tacotron
'''

import re


pad = '_'
punctuations = '!\'(),.:;? '
special = '-'
eos = '~'
letters_en = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
letters_ru = 'АБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзиклмнопрстуфхцчшщъыьэюя'

# Слева - английские буквы, справа - русские. В датасете M-AILABS в некоторых словах русские буквы заменены английскими,
# которые выглядят так же
same_letters_en_ru = {'A': 'А',
                      'B': 'В',
                      'C': 'С',
                      'E': 'Е',
                      'H': 'Н',
                      'K': 'К',
                      'M': 'М',
                      'O': 'О',
                      'P': 'Р',
                      'T': 'Т',
                      'X': 'Х',
                      'a': 'а',
                      'c': 'с',
                      'e': 'е',
                      'o': 'о',
                      'p': 'р',
                      'x': 'х'}

phonemes_en = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
    'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
    'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

# Взято из https://github.com/nsu-ai/russian_g2p/blob/features/russian_g2p/modes/Modern.py
# Самый полный список в https://github.com/nsu-ai/russian_g2p/blob/master/russian_g2p/modes/Phonetics.py
phonemes_ru = [
    'A', 'A0', 'B', 'B0', 'D', 'D0', 'DZ', 'DZ0', 'DZH', 'DZH0', 'E0', 'F', 'F0', 'G',
    'G0', 'GH', 'GH0', 'I', 'I0', 'J0', 'K', 'K0', 'KH', 'KH0', 'L', 'L0', 'M', 'M0', 'N',
    'N0', 'O', 'O0', 'P', 'P0', 'R', 'R0', 'S', 'S0', 'SH', 'SH0', 'T', 'T0', 'TS', 'TS0',
    'TSH', 'TSH0', 'U', 'U0', 'V', 'V0', 'Y', 'Y0', 'Z', 'Z0', 'ZH', 'ZH0'
]

phonemes_set_en = set(phonemes_en)
phonemes_set_ru = set(phonemes_ru)

# Добавление '@' к символам ARPAbet (т.е. к фонемам) для обеспечения их уникальности (некоторые из них совпадают с заглавными буквами)
phonemes_en = ['@' + s for s in phonemes_en]
phonemes_ru = ['@' + s for s in phonemes_ru]

# Списки допустимых символов для модуля MMI
ctc_symbols_en = [pad] + list(letters_en) + phonemes_en #+ [eos]
ctc_symbols_ru = [pad] + list(letters_ru) + phonemes_ru + [eos]

# Списки всех допустимых символов
symbols_en = [pad] + list(special) + list(punctuations) + list(letters_en) + phonemes_en #+ [eos]
symbols_ru = [pad] + list(special) + list(punctuations) + list(letters_ru) + phonemes_ru + [eos]


class CMUDict:
    ''' Простая обёртка для данных CMUDict. http://www.speech.cs.cmu.edu/cgi-bin/cmudict '''
    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='latin-1') as f:
                entries = self.__parse_cmudict(f)
        else:
            entries = self.__parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self.entries = entries
        self.alt_re = re.compile(r'\([0-9]+\)')


    def __len__(self):
        return len(self.entries)


    def __parse_cmudict(self, file):
        cmudict = {}
        for line in file:
            if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
                parts = line.split('  ')
                word = re.sub(self.alt_re, '', parts[0])
                pronunciation = self.__get_pronunciation(parts[1])
                if pronunciation:
                    if word in cmudict:
                        cmudict[word].append(pronunciation)
                    else:
                        cmudict[word] = [pronunciation]
        return cmudict


    def __get_pronunciation(self, s):
        parts = s.strip().split(' ')
        for part in parts:
            if part not in phonemes_set_en:
                return None
        return ' '.join(parts)


    def lookup(self, word):
        ''' Возвращает список из символов ARPAbet (т.е. набор фонем) для данного слова. '''
        return self.entries.get(word.upper())
