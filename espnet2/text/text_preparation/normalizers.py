#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Функции для очистки и нормализации текста перед его отправкой в модель. Выполняется как при обучении, так и при работе с обученной моделью.
Основано на https://github.com/keithito/tacotron

Доступны следующие функции очистки:
    1. 'english_cleaners' - для английского языка
    2. 'russian_cleaners' - для русского языка
    3. 'transliteration_cleaners' - для остальных языков
    4. 'adding_point_at_the_end_of_text' - добавление точки в конец текста, для любого языка
    5. 'adding_eos_at_the_end_of_text' - добавление символа eos в конец текста, для любого языка
'''

import re
from pathlib import Path
import warnings
from unidecode import unidecode

import tensorflow as tf
#from polyglot.transliteration import Transliterator
import polyglot
import accentizer
import stressrnn
from russian_g2p.Accentor import Accentor as g2p_Accentor
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme

from stress_dictionary import StressDictionary

from text_preparation import symbols
from text_preparation.numbers_to_words_en import numbers_to_words_en
from text_preparation.numbers_to_words_ru import numbers_to_words_ru
from text_preparation.expanding_abbreviations_en import expand_abbreviations_en
from text_preparation.expanding_abbreviations_ru import ExpandingAbbreviations_ru
from text_preparation.words_to_alphabetical_transcription_ru import WordsToAlphabeticalTranscription_ru
from text_preparation.mapping_symbols_for_arbitrary_sequences import mapping_symbols_for_arbitrary_sequence
from text_preparation.yo_restorer import YoRestorer
from text_preparation.flow import executor as flow_executor

# Отключение warning уведомлений TensorFlow (предупреждения об устаревших методах и отладочная информация)
# Источник: https://github.com/tensorflow/tensorflow/issues/27023#issuecomment-589673539
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Подавление всех UserWarning от polyglot о том, что он не знает некоторые слова
warnings.filterwarnings('ignore', category=UserWarning)


# Перед использованием нужно выполнить: polyglot download transliteration2.ru
transliterator_en_to_ru = polyglot.transliteration.Transliterator(source_lang='en', target_lang='ru')

expanding_abbreviations_ru = ExpandingAbbreviations_ru()
words_to_alphabetical_transcription_ru = WordsToAlphabeticalTranscription_ru()
yo_restorer_ru = YoRestorer(f_name_add_dict='text_preparation/yo_restorer/additional_yo_dict.txt')

accentizer_from_morpher_ru = accentizer.Accentizer(accentizer.load_standard_dictionaries())
stress_rnn_ru = stressrnn.StressRNN()
g2p_accentor_ru = g2p_Accentor()
g2p_ru = Grapheme2Phoneme()

home = Path(__file__).parent
# Модули ещё сырые!
#expander_abbreviations = ExpandingAbbreviations()
stress_dict = StressDictionary(dict_filenames=[Path(home, 'stress_dictionary', 'special_dictionary_names.json')], dict_names_regex='dictionary_(names|surnames|patronymic)')
# stress_dict = StressDictionary(dict_names_regex='dictionary_(names|surnames|patronymic)')

# text_to_sequence(text, self.language, replacing_symbols=False,
#                  expand_difficult_abbreviations=False,
#                  use_stress_dictionary=use_stress_dictionary,
#                  use_g2p_accentor=use_g2p_accentor,
#                  add_point_at_the_end=add_point_at_the_end)

text = "как твои дела?"

# language = "ru"
# replacing_symbols = False
# expand_difficult_abbreviations = False
# use_stress_dictionary = True
# use_g2p_accentor = True

# normalize_text(text, language, replacing_symbols=replacing_symbols,
#                expand_difficult_abbreviations=expand_difficult_abbreviations,
#                use_stress_dictionary=use_stress_dictionary,
#                use_g2p_accentor=use_g2p_accentor,
#                use_g2p=False,
#                add_point_at_the_end=False).replace(symbols.eos, '')


def transliteration_normalizer(text, **kwargs):
    ''' Очистка и нормализация текста на любом языке, кроме английского и русского, состоит из:
    - удаление нескольких подряд идущих пробелов, замена '.', ',', '!', '?', '(' и ')' на одиночные
    - удаление пробелов перед ',', '.', '!', '?', ':', ';', ')' и после '('
    - замена некоторых знаков препинания на более простые ('...' -> '.', '…' -> '.', '–' -> '-', '‑' -> '-')
    - конвертирование всех символов в строке из Unicode в ASCII

    1. text - строка с текстом для обработки
    2. возвращает обработанный текст '''

    # Базовая очистка текста
    text = basic_cleaning(text)

    # Конвертирование всех символов в строке из Unicode в ASCII (что бы избавиться от разных специфичных символов)
    text = unidecode(text)

    text = text.lower()
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def english_normalizer(text, **kwargs):
    ''' Очистка и нормализация текста на английском языке, состоит из:
    - удаление нескольких подряд идущих пробелов, замена '.', ',', '!', '?', '(' и ')' на одиночные
    - удаление пробелов перед ',', '.', '!', '?', ':', ';', ')' и после '('
    - замена некоторых знаков препинания на более простые ('...' -> '.', '…' -> '.', '–' -> '-', '‑' -> '-')
    - конвертирование всех символов в строке из Unicode в ASCII
    - расшифровка сокращений/аббревиатур по таблице с помощью модуля expanding_abbreviations_en
    - перевод чисел в слова с помощью модуля numbers_to_words_en

    1. text - строка с текстом для обработки
    2. возвращает обработанный текст '''

    # Базовая очистка текста
    text = basic_cleaning(text)

    # Конвертирование всех символов в строке из Unicode в ASCII (что бы избавиться от разных специфичных символов)
    text = unidecode(text)

    text = text.lower()
    text = expand_abbreviations_en(text)
    text = numbers_to_words_en(text)

    text = re.sub(r'\s{2,}', ' ', text)
    return text


def russian_normalizer(text, replacing_symbols=False, expand_difficult_abbreviations=True, use_stress_dictionary=False, use_g2p_accentor=True,
                       use_g2p=True):
    ''' Очистка и нормализация текста на русском языке, состоит из:
    - удаление нескольких подряд идущих пробелов, замена '.', ',', '!', '?', '(' и ')' на одиночные
    - удаление пробелов перед ',', '.', '!', '?', ':', ';', ')' и после '('
    - замена некоторых знаков препинания на более простые ('...' -> '.', '…' -> '.', '–' -> '-', '‑' -> '-')
    - замена символов в тексте на специальные последовательности с помощью модуля replacing_symbols_with_special_sequences
    - расшифровка сокращений/аббревиатур по словарю с помощью модуля expanding_abbreviations_ru
    - транскрипция аббревиатур, инициалов в ФИО и буквенно-цифровых обозначений с помощью модуля words_to_alphabetical_transcription_ru
    - перевод чисел в слова с помощью модуля numbers_to_words_ru
    - транслитерация английских слов в кириллицу
    - восстановление буквы 'ё' с помощью yo_restorer
    - расстановка ударений (можно принудительно задать знаком '+') (не поддерживается для английских слов)
    - перевод слов в фонемы (каждое слово обрамляется '{}', фонемы разделяются пробелами)

    Для транслитерации английских слов в кириллицу используется polyglot.transliteration.Transliterator.
    Для расстановки ударений
    Для перевода слов в фонемы - russian_g2p.Transcription.

    Замена символов в тексте на специальные последовательности позволит корректно и красиво произносить случайные буквенно-цифровые последовательности.
    Например, 'МВУФ02ФОР-3/2703', 'ЛЮСЦЗЖМ-5/2503'.

    ВНИМАНИЕ! Денежные единицы не поддерживаются!

    1. text - строка с текстом для обработки
    2. replacing_symbols - True: заменять символы в тексте на специальные последовательности
    3. expand_difficult_abbreviations - True: расшифровывать сложные в произношении сокращения/аббревиатуры
    4. use_stress_dictionary - True: выполнять расстановку ударений по словарю с ФИО
    5. use_g2p_accentor - True: выполнять расстановку ударений модулем на правилах G2P.Accentor (иногда работает очень медленно)
    6. use_g2p - True: выполнять перевод слов в последовательности фонем с помощью G2P
    7. возвращает обработанный текст '''

    # Очень жирный костыль, нужно исправить при рефакторинге!
    use_unsafe_yo_restorer = use_stress_dictionary

    # Базовая очистка текста
    text = basic_cleaning(text)
    text = flow_executor.run(text)

    # Замена символов в тексте на специальные последовательности
    if replacing_symbols:
        text = mapping_symbols_for_arbitrary_sequence(text)

    # Расшифровка сокращений/аббревиатур
    text = expanding_abbreviations_ru.expand(text, expand_difficult_abbreviations)

    # Транскрипция аббревиатур, инициалов в ФИО и буквенно-цифровых обозначений
    text = words_to_alphabetical_transcription_ru.transcribe(text)

    # Перевод чисел в слова
    text = numbers_to_words_ru(text, remove_spaces_between_numbers=True, account_zeros_at_beginning=True)

    # Разбиение строки на слова и отдельные символы
    words = tokenize(text)

    # Транслитерация английских слов в русский алфавит
    words = transliterator_en_to_ru_wrapper(words)

    # В датасете M-AILABS некоторые буквы в русских словах являются буквами из английского алфавита, которые так же выглядят. Исправление этого
    for i, word in enumerate(words):
        for j, symbol in enumerate(word):
            if symbols.same_letters_en_ru.get(symbol) is not None:
                words[i] = words[i][:j] + symbols.same_letters_en_ru[symbol] + words[i][j+1:]

    # Расстановка части ударений с помощью russian_g2p.Accentor
    if use_g2p_accentor:
        words = g2p_accentor_ru_wrapper(words)

    # Восстановление 'ё' с помощью yo_restorer (идёт после g2p_accentor_ru что бы лишний раз не токенизировать строку)
    text = ''.join(words)
    if use_stress_dictionary:
        text = yo_restorer_ru.restore(text, use_unsafe_dict=use_unsafe_yo_restorer, ignore_word_characteristics=use_unsafe_yo_restorer)

    # Расстановка ударений с помощью Accentizer от Morpher (имеет очень низкий процент неправильно поставленных ударений)
    if use_stress_dictionary:
        text = accentizer_from_morpher_ru_wrapper(text)

    # Расстановка ударений по словарю c помощью stress_dictionary
    if use_stress_dictionary:
        text = stress_dict.stress(text)

    # Расстановка всех оставшихся ударений с помощью RusStress
    if use_stress_dictionary:
        text = stress_rnn_ru.put_stress(text, stress_symbol='+', accuracy_threshold=0.0)

    # Перевод слов в набор фонем с помощью russian_g2p.Grapheme2Phoneme
    if use_g2p:
        words = tokenize(text)
        words = g2p_ru_wrapper(words)
        text = ''.join(words)

    return text


def basic_cleaning(text):
    ''' Базовая очистка текста, состоит из:
    - удаление нескольких подряд идущих пробелов, замена '.', ',', '!', '?', '(' и ')' на одиночные
    - удаление пробелов перед ',', '.', '!', '?', ':', ';', ')' и после '('
    - замена некоторых знаков препинания на более простые ('...' -> '.', '…' -> '.', '–' -> '-', '‑' -> '-')

    1. text - строка с текстом
    2. возвращает обработанный текст '''

    # Удаление символа eos в конце строки, если он есть
    if text and text[-1] == symbols.eos:
        text = text[:-1]

    if text.isspace():
        return text

    # Замена нескольких подряд идущих пробелов, '.', ',', '!', '?', '(' и ')' на одиночные
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\({2,}', '(', text)
    text = re.sub(r'\){2,}', ')', text)

    # Удаление пробелов перед ',', '.', '!', '?', ':', ';', ')' и после '('
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+!', '!', text)
    text = re.sub(r'\s+\?', '?', text)
    text = re.sub(r'\s+\:', ':', text)
    text = re.sub(r'\s+;', ';', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\(\s+', '(', text)

    # Замена некоторых знаков препинания на более простые
    text = text.replace('...', '.')
    text = text.replace('…', '.')
    text = text.replace('–', '-')
    text = text.replace('‑', '-')  # символы имеют разный код

    return text


def tokenize(text):
    ''' Разбиение строки на слова и отдельные символы, и объединение слов, у которых указано ударение символом '+' или которые записаны через '-'.

    1. text - строка с текстом для обработки
    2. возвращает список слов '''

    words = re.split(r'(\W)', text)
    words = [word for word in words if word]

    is_extended_alpha = lambda word: word.replace('+', '').replace('-', '').isalpha()
    i = 1
    while i < len(words) - 1:
        if words[i] == '+' and is_extended_alpha(words[i-1]) and is_extended_alpha(words[i+1]):
            words[i-1] += words[i] + words[i+1]
            del words[i:i+2]
        elif words[i] == '+' and is_extended_alpha(words[i-1]):  # если ударение падает на последнюю букву в слове
            words[i-1] += words[i]
            del words[i:i+1]
        elif words[i] == '-' and is_extended_alpha(words[i-1]) and is_extended_alpha(words[i+1]):
            words[i-1] += words[i] + words[i+1]
            del words[i:i+2]
        else:
            i += 1
    if words and words[-1] == '+':  # если в последнем слове ударение падает на последнюю букву и является последним символом в строке
        words[-2] += words[-1]
        del words[-1]
    return words


def transliterator_en_to_ru_wrapper(words):
    ''' Обёртка для метода polyglot.Transliterator.transliterate() для транслитерации английских слов в русский алфавит.

    ВНИМАНИЕ! polyglot.Transliterator ломается, если слово будет содержать что-либо, кроме английских букв!

    1. words - список слов
    2. возвращает обновлённый список слов '''

    i = 0
    while i < len(words):
        if detect_language(words[i]) == 'en' and words[i].find('-') != -1:
            word_parts = words[i].split('-')
            for j, word_part in enumerate(word_parts):
                transliterated_word = transliterator_en_to_ru.transliterate(word_part.replace('+', ''))
                word_parts[j] = transliterated_word if len(transliterated_word) > 0 else word_parts[j]
            
            # С пробелом звучит лучше (если оставить дефис, будет практически слитно звучать)
            word_parts_with_spaces = []
            for j, word_part in enumerate(word_parts):
                word_parts_with_spaces.append(word_part)
                if j < len(word_parts) - 1:
                    word_parts_with_spaces.append(' ')

            words = words[:i] + word_parts_with_spaces + words[i+1:]
            i += len(word_parts_with_spaces) - 1
        elif detect_language(words[i]) == 'en':
            transliterated_word = transliterator_en_to_ru.transliterate(words[i].replace('+', ''))
            words[i] = transliterated_word if len(transliterated_word) > 0 else words[i]
            i += 1
        else:
            i += 1
    return words


def g2p_accentor_ru_wrapper(words):
    ''' Обёртка для модуля russian_g2p.Accentor. Расставляет ударения знаком '+' после гласной в каждом слове, если знает его. В чистом виде
    имеет очень высокий процент непроставленных ударений, около 65%.

    1. words - список слов
    2. возвращает обновлённый список слов '''

    only_words = []
    mask_for_words = []
    is_extended_alpha = lambda word: word.replace('+', '').replace('-', '').isalpha()
    for word in words:
        if word and is_extended_alpha(word):
            only_words.append(word)
            mask_for_words.append(1)
        else:
            mask_for_words.append(0)

    # Использование PyMorphy2 или RNNMorph для извлечения morphtags никак не влияет на результат, вообще никак
    if only_words:
        only_words = [[word] for word in only_words]
        only_words = g2p_accentor_ru.do_accents(only_words)[0]

    # Со словами 'почем' и 'вагина' происходит баг: g2p_accentor возвращает слово, у которого после каждой буквы стоит символ ударения
    only_words = [word.replace('+', '') if word.count('+') > 2 else word for word in only_words]

    for j, word in enumerate(words):
        if mask_for_words[j]:
            # Восстановление начальной прописной буквы, если она была
            words[j] = only_words[0][0].upper() + only_words[0][1:] if words[j][0].isupper() else only_words[0]
            del only_words[0]
    return words


def g2p_ru_wrapper(words):
    ''' Обёртка для модуля russian_g2p.Grapheme2Phoneme. Выполняет перевод слов в наборы фонем. Для качественного последующего синтеза каждое
    слово должно содержать ударение, поставленное знаком '+' после гласной.

    1. words - список слов
    2. возвращает обновлённый список слов '''

    is_extended_alpha = lambda word: word.replace('+', '').replace('-', '').isalpha()
    for i, word in enumerate(words):
        if word and is_extended_alpha(word):
            transcripted_word = g2p_ru.word_to_phonemes(words[i])
            if len(transcripted_word) != 0:
                words[i] = '{%s}' % ' '.join(transcripted_word)
    return words


def accentizer_from_morpher_ru_wrapper(text, stress_symbol='+'):
    ''' Обёртка для модуля accentizer от morpher.ru. Расставляет ударения знаком '+' после гласной в каждом слове, если знает его. Имеет крайне низкий процент
    неправильно проставленных ударений (~3%), корректно ставит ударения в ФИО не в именительном падеже (особенно в тех, в которых при склонении меняется позиция
    ударения, 'Комисарчу+к' -> 'Комисарчука+') и работает крайне быстро (менее 1 мс на слово).

    1. text - строка с текстом
    2. stress_symbol - символ для обозначения ударения
    3. возвращает обновлённый текст '''

    # Токенизатор (как отдельный, так и встроенный в метод annotate) не понимает символ ударения '+' и разбивает слова на части по этому символу
    # Для исправления этого было добавлено объединение слов по символу ударения после токенизатора с последующим их исключением из обработки
    tokens = list(accentizer.Tokenizer.tokenize(text))
    
    i = 0
    correct_tokens = []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i+1].find(stress_symbol) != -1 and tokens[i+1][-1] == ' ':
            correct_tokens.append(tokens[i]+tokens[i+1].strip())
            correct_tokens.append(' '*tokens[i+1].count(' '))
            i += 2
        elif i + 2 == len(tokens) and tokens[i+1].find(stress_symbol) != -1:
            correct_tokens.append(tokens[i]+tokens[i+1])
            i += 2
        elif i + 2 < len(tokens) and tokens[i+1].find(stress_symbol) != -1 and tokens[i+1][-1] != ' ' and tokens[i][-1].isalpha() and \
                                     tokens[i+2][0].isalpha() and i > 0:
            correct_tokens.append(tokens[i]+tokens[i+1]+tokens[i+2])
            i += 3
        else:
            correct_tokens.append(tokens[i])
            i += 1
    
    # Объединение токенов без ударения между собой. В случае некоторых ФИО это повышает точность расстановки ударений, например:
    # 'Лапшиной Ирины Александровны' - при обработке по токенам неправильное ударение в 'Ирины' ('И+рины' вместо 'Ири+ны')
    i = 0
    while i < len(correct_tokens):
        if correct_tokens[i].find(stress_symbol) == -1 and i + 1 < len(correct_tokens) and correct_tokens[i+1].find(stress_symbol) == -1:
            correct_tokens[i] += correct_tokens[i+1]
            del correct_tokens[i+1]
        else:
            i += 1

    annotated_tokens = []
    for token in correct_tokens:
        if token.find(stress_symbol) == -1:
            annotated_tokens += list(accentizer_from_morpher_ru.annotate(token))
        else:
            annotated_tokens.append(token)

    stressed_tokens = []
    for token in annotated_tokens:
        if isinstance(token, accentizer.AnnotatedToken) and token.annotation:
            stressed_tokens.append(token.annotation.variants[0].apply_to(token.string, stress_symbol, accentizer.StressMarkPlacement.AFTER_STRESSED_VOWEL))
        elif isinstance(token, accentizer.AnnotatedToken):
            stressed_tokens.append(token.string)
        else:
            stressed_tokens.append(token)
    stressed_text = ''.join(stressed_tokens)

    return stressed_text


def adding_point_at_the_end_of_text(text):
    ''' Добавление точки в конец текста, если там нет никаких поддерживаемых знаков препинания. Пробелы и символ eos в конце текста игнорируются.
    Уменьшает вероятность срыва синтеза и в некоторых случаях интонация становится более правильной.

    Список поддерживаемых знаков препинания находится в text_preparation.symbols.punctuations, символ eos в text_preparation.symbols.eos.

    1. text - строка с текстом
    2. возвращает обработанный текст '''

    if not text:
        return text

    # Игнорирование конечных пробелов
    whitespaces_offset = -1
    while len(text) + whitespaces_offset > 0 and text[whitespaces_offset] in [' ', symbols.eos]:
        whitespaces_offset -= 1

    if whitespaces_offset < -1 and text[whitespaces_offset] not in list(symbols.punctuations+symbols.eos):
        text = text[:whitespaces_offset+1] + '.' + text[whitespaces_offset+1:]
    elif whitespaces_offset == -1 and text[whitespaces_offset] not in list(symbols.punctuations+symbols.eos):
        text += '.'
    return text


def adding_eos_at_the_end_of_text(text):
    ''' Добавление символа окончания последовательности (eos) в конец текста, если его там нет. В теории, позволяет уменьшить вероятность срыва синтеза
    и немного упрощает нейронке выучивать окончания фраз, но на практике это не доказано (у кого-то есть положительные изменения от использования eos,
    у кого-то их нет).

    Символ eos находится в text_preparation.symbols.eos.

    1. text - строка с текстом
    2. возвращает обработанный текст '''

    if (text and text[-1] != symbols.eos) or not text:
        text += symbols.eos
    return text


def detect_language(text):
    ''' Определяет, на каком языке написан текст с помощью сравнения каждой буквы с алфавитом конкретного языка. Поддерживает
    только русский и английский язык.

    1. text - строка с текстом
    2. возвращает код языка в виде строки из двух букв (ru для русского языка, en для английского, xx - определить не удалось)

    Так же тестировались langdetect (https://github.com/fedelopez77/langdetect) и langid (https://github.com/saffsd/langid.py).
    При определении языка на основе одного слова они показали очень низкую точность работы. '''

    alphabet_ru = 'абвгдеёжзиклмнопрстуфхцчшщъыьэюя'
    alphabet_en = 'abcdefghijklmnopqrstuvwxyz'

    number_letter_ru = 0
    number_letter_en = 0
    for letter in text.lower():
        if letter in alphabet_ru:
            number_letter_ru += 1
        elif letter in alphabet_en:
            number_letter_en += 1

    if number_letter_ru > number_letter_en:
        return 'ru'
    elif number_letter_en > number_letter_ru:
        return 'en'
    else:
        return 'xx'
