#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для очистки, нормализации и перевода текста в последовательность чисел и обратно.
'''

from text_preparation.text_normalization import normalize_text, LANGUAGES, ctc_symbols, ctc_symbol_to_id, symbol_to_id, id_to_symbol, curly_braces_re
from text_preparation import symbols


def text_to_sequence(text, language='ru', use_text_normalizers=True, replacing_symbols=False, expand_difficult_abbreviations=True,
                     use_stress_dictionary=False, use_g2p_accentor=True, use_g2p=True, add_point_at_the_end=False, add_eos=True):
    ''' Очистка, нормализация и перевод текста в последовательность чисел.

    Замена символов в тексте на специальные последовательности позволит корректно и красиво произносить случайные буквенно-цифровые последовательности.
    Например, 'МВУФ02ФОР-3/2703', 'ЛЮСЦЗЖМ-5/2503'.

    Текст может содержать последовательности символов ARPAbet (фонем), заключенные в фигурные скобки. Например, 'Turn left on {HH AW1 S S T AH0 N} Street'.
    Примечание: набор фонем у каждого языка разный!

    1. text - строка с текстом для обработки
    2. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    3. use_text_normalizers - True: выполнять очистку и нормализацию текста соответствующей указанному языку функцией из normalizers.py
    4. replacing_symbols - True: заменять символы в тексте на специальные последовательности
    5. expand_difficult_abbreviations - True: расшифровывать сложные в произношении сокращения/аббревиатуры
    6. use_stress_dictionary - True: выполнять расстановку ударений по словарю с ФИО
    7. use_g2p_accentor - True: выполнять расстановку ударений модулем на правилах G2P.Accentor (иногда работает очень медленно)
    8. use_g2p - True: выполнять перевод слов в последовательности фонем с помощью G2P
    9. add_point_at_the_end - True: добавлять точку в конец текста, если там нет никаких поддерживаемых знаков препинания (пробелы и eos игнорируются)
    10. add_eos - True: добавлять символ конца последовательности в конец текста, если он поддерживается (используется text_preparation.symbols.eos)
    11. возвращает список из целых чисел, соответствующих символам в тексте '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    # Вход сети не может быть пустым
    if not text:
        text = ' '

    # Проверка наличия фигурных скобок, если они найдены - обработка их содержимого как набор/последовательность фонем
    sequence = []
    while text:
        coincidences = curly_braces_re.match(text)
        if coincidences:  # если найдены фигурные скобки
            normalized_text = normalize_text(coincidences.group(1), language=language,
                                             replacing_symbols=replacing_symbols,
                                             expand_difficult_abbreviations=expand_difficult_abbreviations,
                                             use_stress_dictionary=use_stress_dictionary,
                                             use_g2p_accentor=use_g2p_accentor,
                                             use_g2p=use_g2p,
                                             add_point_at_the_end=False,
                                             add_eos=False) if use_text_normalizers else coincidences.group(1)
            sequence += __symbols_to_sequence(normalized_text, language)

            sequence += __phonemes_to_sequence(coincidences.group(2), language)
            text = coincidences.group(3)
        else:
            normalized_text = normalize_text(text, language=language,
                                             replacing_symbols=replacing_symbols,
                                             expand_difficult_abbreviations=expand_difficult_abbreviations,
                                             use_stress_dictionary=use_stress_dictionary,
                                             use_g2p_accentor=use_g2p_accentor,
                                             use_g2p=use_g2p,
                                             add_point_at_the_end=add_point_at_the_end,
                                             add_eos=add_eos) if use_text_normalizers else text
            sequence += __symbols_to_sequence(normalized_text, language)
            break
    return sequence


def sequence_to_text(sequence, language='ru'):
    ''' Перевод списка из целых чисел, соответствующих символам в тексте, обратно в текст.

    1. sequence - список из целых чисел
    2. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    3. возвращает строку с текстом '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    text = ''
    for symbol_id in sequence:
        if symbol_id in id_to_symbol[language]:
            symbol = id_to_symbol[language][symbol_id]
            if len(symbol) > 1 and symbol[0] == '@':  # оборачивание каждого символа ARPAbet (фонемы) в фигурные скобки
                symbol = '{{{}}}'.format(symbol[1:])
            text += symbol
        else:
            raise AttributeError('Unknown symbol set in sequence!')
    text = text.replace('}{', ' ')
    return text


def __symbols_to_sequence(text, language):
    ''' Перевод очищенного и нормализованного текста в последовательность чисел с учётом возможного представления слов в виде последовательностей фонем.

    1. text - очищенный и нормализованный текст
    2. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    3. возвращает список из целых чисел, соответствующих символам в тексте '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    sequence = []
    while text:
        coincidences = curly_braces_re.match(text)
        if coincidences:
            sequence += [symbol_to_id[language][symbol] for symbol in coincidences.group(1) if symbol in symbol_to_id[language] and symbol is not symbols.pad]
            sequence += __phonemes_to_sequence(coincidences.group(2), language)
            text = coincidences.group(3)
        else:
            sequence += [symbol_to_id[language][symbol] for symbol in text if symbol in symbol_to_id[language] and symbol is not symbols.pad]
            break
    return sequence


def __phonemes_to_sequence(text, language):
    ''' Перевод слова в виде последовательности фонем в последовательность чисел.

    1. text - слово в виде последовательности фонем
    2. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    3. возвращает список из целых чисел, соответствующих фонемам в тексте '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    phonemes_in_text = ['@' + symbol for symbol in text.split()]
    return [symbol_to_id[language][symbol] for symbol in phonemes_in_text if symbol in symbol_to_id[language] and symbol is not symbols.pad]


def get_symbols_length(language):
    ''' Получить длину списка поддерживаемых символов.
    
    1. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    2. возвращает длину списка символов '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    return len(symbol_to_id[language])


def get_ctc_symbols_length(language):
    ''' Получить длину списка допустимых символов для модуля MMI.
    
    1. language - язык текста, поддерживаются русский 'ru', английский 'en' и универсальный 'xx' языки
    2. возвращает длину списка символов '''

    if language not in LANGUAGES:
        raise ValueError("Unsupported language: '{}', supported: 'ru', 'en' and 'xx' languages.".format(language))

    return len(ctc_symbols[language])
