#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для очистки и нормализации текста перед его отправкой в модель. Выполняется как при обучении, так и при работе с обученной моделью.
'''

import re

from text_preparation import normalizers, symbols


LANGUAGES = ['ru', 'en', 'xx']

# Определение функций очистки и нормализации текста для каждого языка
NORMALIZATION_FUNCTIONS = {LANGUAGES[0]: normalizers.russian_normalizer,
                           LANGUAGES[1]: normalizers.english_normalizer,
                           LANGUAGES[2]: normalizers.transliteration_normalizer}

# Определение набора CTC символов для каждого языка
ctc_symbols = {LANGUAGES[0]: symbols.ctc_symbols_ru,
               LANGUAGES[1]: symbols.ctc_symbols_en,
               LANGUAGES[2]: symbols.ctc_symbols_en}

# Отображение символов в числа и обратно для каждого языка
ctc_symbol_to_id = {LANGUAGES[0]: {s: i for i, s in enumerate(symbols.ctc_symbols_ru)},
                    LANGUAGES[1]: {s: i for i, s in enumerate(symbols.ctc_symbols_en)},
                    LANGUAGES[2]: {s: i for i, s in enumerate(symbols.ctc_symbols_en)}}

symbol_to_id = {LANGUAGES[0]: {s: i for i, s in enumerate(symbols.symbols_ru)},
                LANGUAGES[1]: {s: i for i, s in enumerate(symbols.symbols_en)},
                LANGUAGES[2]: {s: i for i, s in enumerate(symbols.symbols_en)}}

id_to_symbol = {LANGUAGES[0]: {i: s for i, s in enumerate(symbols.symbols_ru)},
                LANGUAGES[1]: {i: s for i, s in enumerate(symbols.symbols_en)},
                LANGUAGES[2]: {i: s for i, s in enumerate(symbols.symbols_en)}}

# Регулярное выражение для извлечения текста из фигурных скобок {}
curly_braces_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def normalize_text(text, language='ru', replacing_symbols=False, expand_difficult_abbreviations=True, use_stress_dictionary=False,
                   use_g2p_accentor=True, use_g2p=True, add_point_at_the_end=False, add_eos=True):
    ''' Очистка и нормализация текста для последующего перевода в последовательность чисел.

    Замена символов в тексте на специальные последовательности позволит корректно и красиво произносить случайные буквенно-цифровые последовательности.
    Например, 'МВУФ02ФОР-3/2703', 'ЛЮСЦЗЖМ-5/2503'.

    Текст может содержать последовательности символов ARPAbet (фонем), заключенные в фигурные скобки. Например, 'Turn left on {HH AW1 S S T AH0 N} Street'.
    Примечание: набор фонем у каждого языка разный!

    1. text - строка с текстом для обработки
    2. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    3. replacing_symbols - True: заменять символы в тексте на специальные последовательности
    4. expand_difficult_abbreviations - True: расшифровывать сложные в произношении сокращения/аббревиатуры
    5. use_stress_dictionary - True: выполнять расстановку ударений по словарю с ФИО
    6. use_g2p_accentor - True: выполнять расстановку ударений модулем на правилах G2P.Accentor (иногда работает очень медленно)
    7. use_g2p - True: выполнять перевод слов в последовательности фонем с помощью G2P
    8. add_point_at_the_end - True: добавлять точку в конец текста, если там нет никаких поддерживаемых знаков препинания (пробелы и eos игнорируются)
    9. add_eos - True: добавлять символ конца последовательности в конец текста, если он поддерживается (используется text_preparation.symbols.eos)
    10. возвращает обработанный текст '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    normalizer_func = NORMALIZATION_FUNCTIONS[language]

    # Проверка наличия фигурных скобок, если они найдены - обработка их содержимого как набор/последовательность фонем
    normalized_text = ''
    while text:
        coincidences = curly_braces_re.match(text)
        if coincidences:  # если найдены фигурные скобки
            normalized_text += normalizer_func(coincidences.group(1), replacing_symbols=replacing_symbols,
                                               expand_difficult_abbreviations=expand_difficult_abbreviations,
                                               use_stress_dictionary=use_stress_dictionary,
                                               use_g2p_accentor=use_g2p_accentor,
                                               use_g2p=use_g2p)
            normalized_text += '{{{}}}'.format(coincidences.group(2))
            text = coincidences.group(3)
        else:
            normalized_text += normalizer_func(text, replacing_symbols=replacing_symbols,
                                               expand_difficult_abbreviations=expand_difficult_abbreviations,
                                               use_stress_dictionary=use_stress_dictionary,
                                               use_g2p_accentor=use_g2p_accentor,
                                               use_g2p=use_g2p)
            break

    normalized_text = normalized_text.strip()
    if add_point_at_the_end:
        normalized_text = normalizers.adding_point_at_the_end_of_text(normalized_text)

    if add_eos and symbol_to_id[language].get(symbols.eos):
        normalized_text = normalizers.adding_eos_at_the_end_of_text(normalized_text)

    return normalized_text


def normalize_text_len(normalized_text):
    ''' Подсчёт длины очищенного и нормализованного текста с учётом возможного представления слов в виде последовательностей фонем. Необходимо для
    корректного ограничения допустимой длины текста.

    1. normalized_text - очищенный и нормализованный текст
    2. возвращает корректную длину текста '''

    text_len = 0
    while normalized_text:
        coincidences = curly_braces_re.match(normalized_text)
        if coincidences:
            text_len += len(coincidences.group(1))
            text_len += len(coincidences.group(2).split())
            normalized_text = coincidences.group(3)
        else:
            text_len += len(normalized_text)
            break

    return text_len
