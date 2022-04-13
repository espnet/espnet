#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для перевода чисел в слова. Поддерживаются положительные и отрицательные целые числа и дробные числа до сотых.
Является обёрткой над num2words (https://github.com/savoirfairelinux/num2words).

Функция, реализующая весь функционал: numbers_to_words_ru().

Зависимости: num2words.
'''

import time
import re
from typing import List, Union, Tuple

from num2words import num2words

try:
    from text_preparation.framing_numbers.framing_numbers import NumberWithSpacesCase
    from text_preparation.numerals_declination import NumeralsDeclination_ru
    from text_preparation.numerals_matching import NumeralsMatching_ru
except ModuleNotFoundError:
    from framing_numbers.framing_numbers import NumberWithSpacesCase
    from numerals_declination import NumeralsDeclination_ru
    from numerals_matching import NumeralsMatching_ru


framing_numbers_with_spaces = NumberWithSpacesCase()
numerals_declination_ru = NumeralsDeclination_ru()
numerals_matching_ru = NumeralsMatching_ru()


def string_to_number(number_as_str, account_zeros_at_beginning=True):
    ''' Конвертирование числа в виде строки в число int или float. Поддерживаются следущие представления чисел:
    - с разделением классов точкой и без дробной части, например 12.345.678 (должно быть 2 и более точек)
    - с разделением классов запятой и без дробной части, например 12,345,678 (должно быть 2 и более запятых)
    - с разделением классов точкой (или без разделения классов) и разделением дробной части запятой, например 12.345,67 (должна быть только 1 запятая)
    - с разделением классов запятой (или без разделения классов) и разделением дробной части точкой, например 12,345.67 (должна быть только 1 точка)
    - без разделения классов и дробных, например 1234567

    Отрицательные числа должны иметь в самом начале знак '-' и отделяться от предыдущего слова пробелом или любой не буквой.

    Дополнительно выполняется нормализация полученной строки: проверка наличия в ней цифр; отделение всех символов, которые не являются цифрами и
    находятся перед и после числа (за исключением знака минус перед числом и случаев вида "текст-число" (например "Восточная-38")); удаление всех
    символов между первой и последней цифрой, кроме самих цифр, точек и запятых; удаление нескольких подряд идущих точек и запятых внутри числа.

    ВНИМАНИЕ! Если между цифрами в переданном числе есть какие-либо символы, кроме точек и запятых - они будут удалены!

    1. number_as_str - число в виде строки
    2. account_zeros_at_beginning - True: учёт нулей в начале числа, т.е. если число начинается с 0 - добавление к символам перед числом слов "ноль"
    3. возвращает tuple из числа int или float и списка из двух строк со всеми символами, найденными перед и после числа:
        (int/float, ['symbols_before_number', 'symbols_after_number']) '''

    if not isinstance(number_as_str, str):
        raise TypeError("'number_as_str' must be a string")

    if len(re.sub(r'\D', '', number_as_str)) == 0:
        raise ValueError("'number_as_str' does not contain numbers")

    if len(number_as_str) == 0:
        raise ValueError("'number_as_str' is empty string")

    # Нормализация числа: поиск индексов начальной и конечной цифры, что бы отбросить всё, что не является числом и "прилипло" к нему
    start_index = 0
    end_index = 0
    for i, symbol in enumerate(number_as_str):
        if symbol.isdigit():
            start_index = i
            if i != 0 and number_as_str[i-1] == '-':
                start_index -= 1
            break
    for i, symbol in enumerate(reversed(number_as_str)):
        if symbol.isdigit():
            end_index = len(number_as_str) - i
            break
    extra_characters = [number_as_str[:start_index], number_as_str[end_index:]]
    number_as_str = number_as_str[start_index:end_index]

    # Нормализация числа: учёт возможного написания числа в формате "текст-число", например "Восточная-38" или "Учреждение-10"
    if extra_characters[0] and extra_characters[0][-1].isalpha():
        extra_characters[0] += number_as_str[:1]
        number_as_str = number_as_str[1:]

    # Нормализация числа: учёт возможного знака минус в начале
    is_negative = False
    if number_as_str[0] == '-':
        is_negative = True
        number_as_str = number_as_str[1:]

    # Нормализация числа: удаление всех символов, кроме цифр, точек и запятых, которые могли остаться между цифрами, и удаление нескольких подряд
    # идущих точек и запятых
    number_as_str = re.sub(r'[^0-9\.,]', '', number_as_str)
    number_as_str = re.sub(r'\.{2,}', '.', number_as_str)
    number_as_str = re.sub(r',{2,}', ',', number_as_str)

    # Нормализация числа: учёт нулей в начале числа, если число начинается с 0 - добавление к дополнительным символам перед числом слов "ноль"
    if account_zeros_at_beginning:
        zeros_as_list = []
        for symbol in number_as_str:
            if symbol == '0':
                zeros_as_list.append('ноль')
            else:
                break
        if len(zeros_as_list) > 0:
            if len(zeros_as_list) == len(number_as_str):  # если всё число состоит из нулей - избегание появления "дополнительного" нуля
                del zeros_as_list[-1]
            zeros_as_str = ' '.join(zeros_as_list) + ' '
            if len(extra_characters[0]) > 0:
                extra_characters[0] += ' ' + zeros_as_str
            else:
                extra_characters[0] = zeros_as_str

    # Подсчёт количества точек и запятых
    number_of_points = 0
    number_of_commas = 0
    for symbol in number_as_str:
        if symbol == '.':
            number_of_points += 1
        elif symbol == ',':
            number_of_commas += 1

    # Число с разделением классов точкой и без дробной части, например 12.345.678 (должно быть 2 и более точек)
    if number_of_points > 1 and number_of_commas == 0:
        number_as_str = number_as_str.replace('.', '')
        number = int(number_as_str) * -1 if is_negative else int(number_as_str)
        return number, extra_characters
        
    # Число с разделением классов запятой и без дробной части, например 12,345,678 (должно быть 2 и более запятых)
    if number_of_commas > 1 and number_of_points == 0:
        number_as_str = number_as_str.replace(',', '')
        number = int(number_as_str) * -1 if is_negative else int(number_as_str)
        return number, extra_characters
    
    # Число без разделения классов, например 12345678
    if number_of_commas == 0 and number_of_points == 0:
        number = int(number_as_str) * -1 if is_negative else int(number_as_str)
        return number, extra_characters

    # Число с разделением классов точкой (или без разделения классов) и разделением дробных запятой, например 12.345,67 (должна быть только 1 запятая)
    if number_as_str.rfind(',') != -1 and number_as_str.find(',') == number_as_str.rfind(','):
        number_as_str_integer = number_as_str[:number_as_str.find(',')]
        number_as_str_integer = number_as_str_integer.replace('.', '')
        number_as_str = number_as_str_integer + number_as_str[number_as_str.rfind(','):]
    
    # Число с разделением классов запятой (или без разделения классов) и разделением дробных точкой, например 12,345.67 (должна быть только 1 точка)
    if number_as_str.rfind('.') != -1 and number_as_str.find('.') == number_as_str.rfind('.'):
        number_as_str_integer = number_as_str[:number_as_str.find('.')]
        number_as_str_integer = number_as_str_integer.replace(',', '')
        number_as_str = number_as_str_integer + number_as_str[number_as_str.rfind('.'):]
    
    # Замена запятых на точки, отделение дробной части последней точкой и удаление остальных точек в числе
    number_as_str = number_as_str.replace(',', '.')
    if number_as_str.rfind('.') != -1:
        number_as_str_integer = number_as_str[:number_as_str.rfind('.')]
        number_as_str_integer = number_as_str_integer.replace('.', '')
        number_as_str = number_as_str_integer + number_as_str[number_as_str.rfind('.'):]
    
    number = float(number_as_str) * -1 if is_negative else float(number_as_str)
    return number, extra_characters


def one_number_to_words_ru(number, account_zeros_at_beginning=True) -> Tuple[Union[int, float], str]:
    ''' Перевод одного числа в слова. Перевод осуществляется пакетом num2words, данная функция выполняет подготовку полученного числа и
    косметическую коррекцию результата.

    Подготовка числа состоит из: конвертирования строки в число int или float (если требуется); округление до сотых для float; проверка допустимого
    количества знаков в числе и преобразование float в int, если дробная часть равна 0.

    Коррекция результата затрагивает только дробные числа: замена слова 'запятая' на 'целая(-ых)' и добавление в конце
    'десятая(-ых)'/'сотая(-ых)'.

    Поддерживаются следующие числа, записанные в виде строки:
    - с разделением классов точкой и без дробной части, например 12.345.678 (должно быть 2 и более точек)
    - с разделением классов запятой и без дробной части, например 12,345,678 (должно быть 2 и более запятых)
    - с разделением классов точкой (или без разделения классов) и разделением дробной части запятой, например 12.345,67 (должна быть только 1 запятая)
    - с разделением классов запятой (или без разделения классов) и разделением дробной части точкой, например 12,345.67 (должна быть только 1 точка)
    - без разделения классов и дробных, например 1234567

    Отрицательные числа должны иметь в самом начале знак '-' и отделяться от предыдущего слова пробелом или любой не буквой.

    Дополнительно выполняется нормализация полученной строки: проверка наличия в ней цифр; отделение всех символов, которые не являются цифрами и
    находятся перед и после числа (за исключением знака минус перед числом и случаев вида "текст-число" (например "Восточная-38")); удаление всех
    символов между первой и последней цифрой, кроме самих цифр, точек и запятых; удаление нескольких подряд идущих точек и запятых внутри числа.

    ВНИМАНИЕ! Если между цифрами в переданном числе есть какие-либо символы, кроме точек и запятых - они будут удалены! Так же числа, записанные через
    пробел (например 12 345), не поддерживаются!

    1. number - число в виде строки, int или float
    2. account_zeros_at_beginning - True: учёт нулей в начале числа, т.е. если число начинается с 0 - добавление к символам перед числом слов "ноль"
    3. возвращает tuple из числа в виде int/float и числа в виде последовательности слов в строке '''

    if isinstance(number, str):
        number, extra_characters = string_to_number(number, account_zeros_at_beginning)
    else:
        extra_characters = '', ''
    number = round(number, 2)

    # Проверка количества знаков в числе (num2words поддерживает числа до 999 нониллионов, т.е. 33 знака или 16 знаков для float)
    number_as_str = str(abs(number))
    # Если число float в целой части имеет больше 16 цифр - оно представляется в "научной нотации" (через e+XX или e-XX), при этом все остальные цифры
    # и дробная часть теряется. Так представляются большие дробные числа в Python 3.7. Конвертирование такого числа в число int с 16 знаками
    if number_as_str.rfind('e') != -1:
        number_as_str = '{:.0f}'.format(number)
        number = int(number_as_str[:16])
    # Если количество знаков в целом числе больше 33 - отбрасываем всё остальное, иначе будет ошибка (num2words поддерживает числа до 999 нониллионов)
    elif len(number_as_str) > 33:
        number = int(number_as_str[:33])

    # Если float и на конце 0, то преобразуем в int
    if isinstance(number, float):
        number_as_str = str(abs(number))
        number_as_str_fraction = number_as_str[number_as_str.rfind('.')+1:]
        if number_as_str_fraction == '0':
            number = int(number)

    number_as_text = num2words(number, lang='ru')

    if number_as_text.rfind('запятая') != -1:
        # Коррекция целой части
        integer_part = number_as_text[:number_as_text.rfind('запятая')]
        integer_part = integer_part.split(' ')
        integer_part = [word for word in integer_part if word.strip()]

        if integer_part[-1] == 'один':
            integer_part[-1] = 'одна'
        if integer_part[-1] == 'два':
            integer_part[-1] = 'две'

        if integer_part[-1] == 'одна':
            integer_part.append('целая')
        else:
            integer_part.append('целых')

        # Коррекция дробной части
        # Исправление очень специфичной ситуации: что бы правильно переводилась дробная часть с сотыми от 1 до 9 (т.е. .01, .08 и т.д.), без исправления
        # такие сотые превращаются в десятые
        number_as_str = str(abs(number))
        number_as_str_fraction = number_as_str[number_as_str.rfind('.')+1:]
        if number_as_str_fraction[0] == '0':
            number_as_text = number_as_text[:number_as_text.rfind('запятая')+8] + 'ноль' + number_as_text[number_as_text.rfind('запятая')+7:]

        fractional_part = number_as_text[number_as_text.rfind('запятая')+8:]
        fractional_part = fractional_part.split(' ')

        if fractional_part[-1] == 'один':
            fractional_part[-1] = 'одна'
        if fractional_part[-1] == 'два':
            fractional_part[-1] = 'две'

        numbers_11_to_19 = ['одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 'пятнадцать', 'шестнадцать', 'семнадцать', 'восемнадцать', 'девятнадцать']
        if len(fractional_part) == 1 and fractional_part[-1] not in numbers_11_to_19 and fractional_part[-1] == 'одна':
            fractional_part.append('десятая')
        elif len(fractional_part) == 1 and fractional_part[-1] not in numbers_11_to_19:
            fractional_part.append('десятых')
        elif fractional_part[-1] == 'одна':
            fractional_part.append('сотая')
        elif len(fractional_part) < 3:
            fractional_part.append('сотых')

        if 'ноль' in fractional_part:
            fractional_part.remove('ноль')

        number_as_text = ' '.join(integer_part) + ' ' + ' '.join(fractional_part)

    number_as_text = extra_characters[0] + number_as_text + extra_characters[1]
    return number, number_as_text


def numbers_to_words_ru(text, remove_spaces_between_numbers=True, account_zeros_at_beginning=True):
    ''' Поиск чисел в строке и перевод их в слова. Перевод осуществляется пакетом num2words и функциями one_number_to_words_ru() и string_to_number() для
    подготовки найденных чисел и косметической коррекции результата.

    Подготовка строки состоит из: удаление пробелов между любыми двумя подряд идущими числами (например 12 345.67, используется когда количество пробелов
    между числами меньше 70% от общего числа пробелов); замена дефисов и двоеточий между числами на пробелы (так лучше звучит итоговая синтезированная речь)
    и обрамление всех чисел пробелами (кроме случаев, когда перед числом есть символ '(' и/или после числа есть символы ',', '.', '!', '?', ':', ';', ')').

    Подготовка каждого найденного числа состоит из (функция one_number_to_words_ru()): конвертирования строки с числом в int или float (если требуется);
    округление до сотых для float; проверка допустимого количества знаков в числе и преобразование float в int, если дробная часть равна 0.

    Косметическая коррекция результата затрагивает только дробные числа (функция one_number_to_words_ru()): замена слова 'запятая' на 'целая(-ых)' и
    добавление в конце 'десятая(-ых)'/'сотая(-ых)'.

    Поддерживаются следующие числа, записанные в виде строки (функция string_to_number()):
    - с разделением классов точкой и без дробной части, например 12.345.678 (должно быть 2 и более точек)
    - с разделением классов запятой и без дробной части, например 12,345,678 (должно быть 2 и более запятых)
    - с разделением классов точкой (или без разделения классов) и разделением дробной части запятой, например 12.345,67 (должна быть только 1 запятая)
    - с разделением классов запятой (или без разделения классов) и разделением дробной части точкой, например 12,345.67 (должна быть только 1 точка)
    - с разделением классов пробелами, например 12 345.678
    - без разделения классов и дробных, например 1234567

    Отрицательные числа должны иметь в самом начале знак '-' и отделяться от предыдущего слова пробелом или любой не буквой.

    Дополнительно выполняется нормализация полученной строки: проверка наличия в ней цифр; отделение всех символов, которые не являются цифрами и
    находятся перед и после числа (за исключением знака минус перед числом и случаев вида "текст-число" (например "Восточная-38")); удаление всех
    символов между первой и последней цифрой, кроме самих цифр, точек и запятых; удаление нескольких подряд идущих точек и запятых внутри числа.

    ВНИМАНИЕ! Если между цифрами в найденных числах есть какие-либо символы, кроме точек и запятых - они будут удалены!

    1. text - строка с числами или число int или float
    2. remove_spaces_between_numbers - True: удаление пробелов между любыми двумя подряд идущими числами (с некоторым условием, см. выше)
    3. account_zeros_at_beginning - True: учёт нулей в начале числа, т.е. если число начинается с 0 - замена их на слова "ноль"
    4. возвращает text с числами, переведёнными в слова '''

    if isinstance(text, int) or isinstance(text, float):
        _, number_as_text = one_number_to_words_ru(text, account_zeros_at_beginning)
        return number_as_text

    if not text or text.isspace():
        return text

    # Своеобразный костыль: когда в тексте много чисел, записанных через пробел - отключение удаления пробелов между числами. Подсчитывает
    # количество пробелов между числами и общее количество пробелов в тексте. Если количество пробелов между числами больше 70% от общего
    # числа пробелов - удаление пробелов между числами отключается (число 70% взято "на глаз"). Это сделано для корректной обработки текста,
    # содержащего очень много цифр, записанных через пробел (например, '1 2 3 4 5 6 7 8 9 10 100 200 300 и т.д.')
    number_coincidences = len(re.findall(r'(?<=\d)\s+(?=\d)', text))
    number_spaces = len([symbol for symbol in text if symbol == ' '])
    if number_spaces > 0 and number_coincidences / number_spaces <= 0.7 and remove_spaces_between_numbers:
        # Удаление пробелов между двумя подряд идущими числами (например 12 345.67), источник: https://issue.life/questions/54886798
        # Используются шаблоны positive lookbehind assertion и lookahead assertion, их описание: https://habr.com/ru/post/349860/
        text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)


    # Сохранение начальных и конечных пробелов (необходимо для корректной работы перевода текста с фонемами в последовательность чисел и обратно)
    # Циклы работают быстрее регулярки r'\s+' примерно в 300 раз (регулярка работает за 0.00249290, циклы за 0.00000715)
    #extra_whitespaces = [re.match(r'\s+', text).group(0) if re.match(r'\s+', text) else '',
    #                     re.match(r'\s+', text[::-1]).group(0) if re.match(r'\s+', text[::-1]) else '']
    extra_whitespaces = ['', '']
    for symbol in text:
        if symbol.isspace():
            extra_whitespaces[0] += symbol
        else:
            break

    for symbol in text[::-1]:
        if symbol.isspace():
            extra_whitespaces[1] += symbol
        else:
            break


    # Обработка графиков работы/временных промежутков, например: '09:00-17:00' => 'с 09:00 до 17:00'
    text = re.sub(r'(\d\d\:\d\d)-(\d\d\:\d\d)', r'с \1 до \2', text)
    match = re.search(r'с\s\d\d\:\d\d\sдо\s\d\d\:\d\d', text)
    while match:
        work_schedule = match.group(0)
        work_schedule_indexes = match.regs[0]

        work_schedule = work_schedule.replace(':', ' ')

        words = work_schedule.split(' ')
        is_minutes = False
        for i, word in enumerate(words):
            if len(re.sub(r'\D', '', word)) > 0:
                _, words[i] = one_number_to_words_ru(word, account_zeros_at_beginning=is_minutes)
                is_minutes = True
            else:
                is_minutes = False
        work_schedule = ' '.join(words)

        work_schedule_p1 = work_schedule[work_schedule.find('с ')+2:work_schedule.find(' до ')]
        work_schedule_p1 = numerals_matching_ru.make_agree_with_preposition('с', work_schedule_p1)

        work_schedule_p2 = work_schedule[work_schedule.find(' до ')+4:]
        work_schedule_p2 = numerals_matching_ru.make_agree_with_preposition('до', work_schedule_p2)
        work_schedule = 'с {} до {}'.format(work_schedule_p1, work_schedule_p2)

        # Добавление "паузы" после каждого временного промежутка путём замены запятой на точку (для других знаков препинания "пауза"
        # по умолчанию достаточная)
        if match.regs[0][1] < len(text) and text[match.regs[0][1]] == ',':
            text = text[:match.regs[0][1]] + '.' + text[match.regs[0][1]+1:]

            offset_to_next_letter = 1
            while match.regs[0][1] + offset_to_next_letter < len(text):
                if text[match.regs[0][1]+offset_to_next_letter].isalpha():
                    text = text[:match.regs[0][1]+offset_to_next_letter] + text[match.regs[0][1]+offset_to_next_letter].upper() + \
                           text[match.regs[0][1]+offset_to_next_letter+1:]
                    break
                else:
                    offset_to_next_letter += 1

        text = text[:match.regs[0][0]] + work_schedule + text[match.regs[0][1]:]
        match = re.search(r'с\s\d\d\:\d\d\sдо\s\d\d\:\d\d', text)


    # Замена дефисов и двоеточий между числами на пробелы (не касается одиночных чисел, начинающихся с дефиса) (так правильнее синтезируется, чем если
    # просто оставлять дефиз как есть - будут неестественно длинные паузы)
    text = re.sub(r'(?<=\d)-(?=\d)|(?<=\d)\:(?=\d)', ' ', text)


    # Обрамление всех чисел пробелами с учётом возможного знака минус перед числом (т.е. дефиса), слитного дефиса между словом и числом (случаи вида
    # "текст-число", например "Восточная-38") и разделителей разрядов и дробной части (т.е. с учётом точек и запятых) (кроме случаев, когда перед числом
    # есть символ '(' и/или после числа есть символы ',', '.', '!', '?', ':', ';', ')')
    #text = re.sub(r'(([а-яёА-ЯЁa-zA-Z\-]*\-|\()?\d+([\.,]?(\-?:?\d)*)+[\.,!\?:;\)]*)', r' \1 ', text)
    #text = re.sub(r'\s{2,}', ' ', text)

    text = framing_numbers_with_spaces.frame_numbers(text)
    text = text.strip()


    # Разбиение строки по пробелам, поиск и перевод всех чисел в слова
    words = text.split(' ')
    for i, word in enumerate(words):
        if len(re.sub(r'\D', '', word)) > 0:
            offset = 0
            while i + offset < len(words) and offset < 3:
                offset += 1

            try:
                number_digit, number_as_text = one_number_to_words_ru(word, account_zeros_at_beginning)
                number_as_text = numerals_declination_ru.decline(number_as_text, word)
                words[i:i+offset] = numerals_matching_ru.make_agree(number_digit, words[i+1:i+offset], number_as_text)
                if i > 0:
                    words[i] = numerals_matching_ru.make_agree_with_preposition(words[i-1], words[i])
            except Exception as e:
                print(e)
                pass

    text = extra_whitespaces[0] + ' '.join(words) + extra_whitespaces[1]
    return text


# Особенности:
# 1. Нет поддержки денежных единиц
# 2. Не ставит заглавные буквы в случае, если передана строка и там несколько предложений (или число стоит в начале предложения)


def main():
    test_texts = {
        '': '',
        ' ': ' ',
        'Вт-Сб 11:05-22:00, перерыв: 03:30-04:15':
                'Вт-Сб с одиннадцати ноль пяти до двадцати двух ноль ноль. Перерыв: с трёх тридцати до четырёх пятнадцати',
        '11 до 223 и со 172':
                'одиннадцать до двухсот двадцати трёх и со сто семидесяти двух',
        'Лечебно-Исправительное-Учреждение-10, мой дом на улице-201, уран-238, гелий-3,-98 тебе25 лет':
                'Лечебно-Исправительное-Учреждение-десять, мой дом на улице-двести один, уран-двести тридцать восемь, гелий-три, ' + \
                'минус девяносто восемь тебе двадцать пять лет',
        '92345678901234567.89 и 92345678901234567890123456789012345678':
                'девять квадриллионов двести тридцать четыре триллиона пятьсот шестьдесят семь миллиардов восемьсот девяносто миллионов ' + \
                'сто двадцать три тысячи четыреста пятьдесят шесть и девятьсот двадцать три нониллиона четыреста пятьдесят шесть октиллионов ' + \
                'семьсот восемьдесят девять септиллионов двенадцать секстиллионов триста сорок пять квинтиллионов шестьсот семьдесят восемь ' + \
                'квадриллионов девятьсот один триллион двести тридцать четыре миллиарда пятьсот шестьдесят семь миллионов восемьсот девяносто ' + \
                'тысяч сто двадцать три',
        '1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 1000 2000 3000 4000 и т.д.':
                'один два три четыре пять шесть семь восемь девять десять двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят ' + \
                'девяносто сто одна тысяча две тысячи три тысячи четыре тысячи и т.д.',
        ' Привет: -6 -811- 34-34 и -23 ребро  222,43  4,- о да детка! ':
                ' Привет: минус шесть минус восемьсот одиннадцать - тридцать четыре тридцать четыре и минус двадцать три ребра двести двадцать ' + \
                'две целых сорок три сотых, - о да детка! ',
        '.м23, 4-ыло-222,-=хоп-43-  *4-,-руб 50руб о да детка 56! (12 и 39) ,213, 32. 324, 234?32,05, 98:90:09, 9-05-2, 3-4, 43; 4;2':
                '. м2три, четыре-ыло минус двести двадцать два, -= хоп-сорок три -  * четыре -,-руб пятьдесят руб о да детка ' + \
                'пятьдесят шесть! (двенадцать и тридцать девять) , двести тринадцать, тридцать два. триста двадцать четыре, двести тридцать ' + \
                'четыре? тридцать две целых пять сотых, девяносто восемь девяносто ноль девять, девять ноль пять два, три четыре, сорок три; ' + \
                'четыре; два',
        '12 345,90, -42,351.01, 42.356,34, 42,356,342, 42.356.342,..m103..,,,.400.uhv.,mn,,...':
                'двенадцать тысяч триста сорок пять целых девять десятых, минус сорок две тысячи триста пятьдесят одна целая одна сотая, сорок ' + \
                'две тысячи триста пятьдесят шесть целых тридцать четыре сотых, сорок два миллиона триста пятьдесят шесть тысяч триста сорок ' + \
                'два, сорок два миллиона триста пятьдесят шесть тысяч триста сорок два, .. m1 ноль три. .,,,. четыреста. uhv.,mn,,...',
        1234562.1:
                'один миллион двести тридцать четыре тысячи пятьсот шестьдесят две целых одна десятая',
        'ну, это полный 3,14здец я вам скажу':
                'ну, это полный три целых четырнадцать сотых здеца я вам скажу',
        'Да ну, 28 934 587,5рубль?':
                'Да ну, двадцать восемь миллионов девятьсот тридцать четыре тысячи пятьсот восемьдесят семь целых пять десятых рублей?',
        '28 000 000 304 5 6 и т.д.':
                'двадцать восемь ноль ноль ноль ноль ноль ноль триста четыре пять шесть и т.д.',
        '99 000 000 рубль':
                'девяносто девять миллионов рублей',
        'дом 89 строение3 корпус81 помещение23':
                'дом восемьдесят девять строение три корпус восемьдесят один помещение двадцать три',
        'И дал он ему смачную пощёчину...':
                'И дал он ему смачную пощёчину...',
        'Балет с 2 лет':
                'Балет с двух лет',
        '35-летия победы':
                'тридцати пяти-летия победы',
        '01-98, 03-31ого затем 31-я в 45:93:34 и 09-98-43-я':
                'ноль один девяносто восемь, ноль три тридцать первого затем тридцать первая в сорок пять девяносто три тридцать четыре и ноль девять ' + \
                'девяносто восемь сорок третья',
        'Дом 45 Э корпус 90 ю офис 66 я. Дом 45 Э, корпус 90 ю, офис 66 я.':
                'Дом сорок пять э корпус девяносто ю офис шестьдесят шесть я. Дом сорок пять э, корпус девяносто ю, офис шестьдесят шесть я.',
    }


    print('[i] Тестирование на {} примерах...'.format(len(test_texts)))
    elapsed_times = []
    error_result_tests = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        result = numbers_to_words_ru(text)
        elapsed_times.append(time.time()-start_time)

        is_correct = True
        if result != test_texts[text]:
            is_correct = False
            error_result_tests.append("{}. '{}' ->\n\t'{}'".format(i+1, text, result))

        print("{}. '{}' ->\n\t'{}', {}, {:.6f} с".format(i+1, text, result, str(is_correct).upper(), elapsed_times[-1]))
    print('[i] Среднее время обработки {:.6f} с или {:.2f} мс'.format(sum(elapsed_times)/len(elapsed_times), sum(elapsed_times)/len(elapsed_times)*1000))

    if error_result_tests:
        print('\n[E] Ошибки в следующих примерах:')
        for error_result in error_result_tests:
            print(error_result)
    else:
        print('\n[i] Ошибок не обнаружено')


    # Тестирование на вводе пользователя
    while True:
        text = input('\n[i] Введите текст: ')
        start_time = time.time()
        prepared_text = numbers_to_words_ru(text)
        elapsed_time = time.time() - start_time

        print("[i] Результат: '{}'".format(prepared_text))
        print('[i] Время обработки {:.6f} с или {:.2f} мс'.format(elapsed_time, elapsed_time*1000))


if __name__ == '__main__':
    main()
