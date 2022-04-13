#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для склонения числа в соответствии с его окончанием.

Зависимости: pymorphy2[fast]
'''

import re
import time
from pymorphy2 import MorphAnalyzer


class NumeralsDeclination_ru:
    ''' Склонение числа в соответствии с его окончанием. Для склонения используется PyMorphy2. '''

    # Падежи в PyMorphy2:
    #   nomn - именительный
    #   gent - родительный
    #   datv - дательный
    #   accs - винительный
    #   ablt - творительный
    #   loct - предложный
    # Род в PyMorphy2:
    #   masc - мужской
    #   femn - женский
    #   neut - средний
    dict_of_ordinal_endings_and_pymorphy_tags = {
        'ого': {'gent'},
        'ому': {'datv'},
        'ым': {'ablt'},
        'ом': {'loct'},
        'е': {'neut', 'nomn'},
        'й': {'masc', 'nomn'},
        'я': {'femn', 'nomn'},
    }

    dict_of_stored_endings_and_pymorphy_tags = {
        'летия': {'gent'},  # '75-летия победы'
    }

    dict_of_ordinal_numbers = {
        '0': 'нулевой',
        '1': 'первый',
        '2': 'второй',
        '3': 'третий',
        '4': 'четвёртый',
        '5': 'пятый',
        '6': 'шестой',
        '7': 'седьмой',
        '8': 'восьмой',
        '9': 'девятый',
        '10': 'десятый',
        '11': 'одиннадцатый',
        '12': 'двенадцатый',
        '13': 'тринадцатый',
        '14': 'четырнадцатый',
        '15': 'пятнадцатый',
        '16': 'шестнадцатый',
        '17': 'семнадцатый',
        '18': 'восемнадцатый',
        '19': 'девятнадцатый',
        '20': 'двадцатый',
        '30': 'тридцатый',
        '40': 'сороковой',
        '50': 'пятидесятый',
        '60': 'шестидесятый',
        '70': 'семидесятый',
        '80': 'восьмидесятый',
        '90': 'девяностый',
        '100': 'сотый',
        '200': 'двухсотый',
        '300': 'трехсотый',
        '400': 'четырехсотый',
        '500': 'пятисотый',
        '600': 'шестисотый',
        '700': 'семисотый',
        '800': 'восьмисотый',
        '900': 'девятисотый',
        '1000': 'тысячный',
        '1000000': 'миллионный',
        '1000000000': 'миллиардный',
        '1000000000000': 'триллионный'
    }

    def __init__(self):
        self.morph_analyzer = MorphAnalyzer()
        self.word_search_re = re.compile(r'([\W\d_])')
    

    def __tokenize(self, text, ignore_hyphen=False):
        ''' Разбиение строки на слова и отдельные символы, и объединение слов, у которых указано ударение символом '+' или которые записаны через '-'.

        1. text - строка с текстом для обработки
        2. ignore_hyphen - True: разделять слова, записанные через дефис
        3. возвращает список слов '''

        words = self.word_search_re.split(text)
        words = [word for word in words if word]

        is_extended_alpha = lambda word: word.replace('+', '').replace('-', '').isalpha()
        i = 1
        while i < len(words) - 1:
            # Если символ '+' не в начале/конце слова (т.е. предыдущее и следующее слово являются последовательностями букв)
            if words[i] == '+' and is_extended_alpha(words[i-1]) and is_extended_alpha(words[i+1]):
                words[i-1] += words[i] + words[i+1]  # объединение предыдущего, текущего и следующего слова
                del words[i:i+2]  # удаление текущего и следующего слова

            # Если символ '+' в конце слова (т.е. предыдущее слово является последовательностью букв)
            elif words[i] == '+' and is_extended_alpha(words[i-1]):
                words[i-1] += words[i]  # объединение предыдущего и текущего слова
                del words[i:i+1]  # удаление текущего слова

            # Если символ '-' не в начале/конце слова (т.е. предыдущее и следующее слово являются последовательностями букв)
            elif not ignore_hyphen and words[i] == '-' and is_extended_alpha(words[i-1]) and is_extended_alpha(words[i+1]):
                words[i-1] += words[i] + words[i+1]  # объединение предыдущего, текущего и следующего слова
                del words[i:i+2]  # удаление текущего и следующего слова

            else:
                i += 1

        # Если символ '+' в конце последнего слова (т.е. предыдущее слово является последовательностью букв)
        if words and len(words) > 1 and words[-1] == '+' and is_extended_alpha(words[-2]):  
            words[-2] += words[-1]
            del words[-1]
        return words


    def __search_ordinal_number_for_declension(self, replace_with_ordinal, number_as_digit):
        ''' Поиск порядкового числа для склонения. По последним цифрам в числе ищет совпадения со словарём порядковых чисел и возвращает найденное
        порядковое число, записанное словом. Например, '305' -> 'пятый'.
        
        1. replace_with_ordinal - True: заменять число на порядковое при склонении
        2. number_as_digit - число, записанное цифрами, в виде строки
        3. возвращает найденное порядковое число или None '''

        ordinal_for_declension = None
        if replace_with_ordinal and self.dict_of_ordinal_numbers.get(number_as_digit):
            ordinal_for_declension = self.dict_of_ordinal_numbers[number_as_digit]

        elif replace_with_ordinal:
            # Числа 2, 1, 3, 4, 7, 10, 13 являются минимальной длиной числа для поиска N его последних цифр в словаре порядковых чисел. Длина перечислена в
            # таком порядке, что бы не было ложных совпадений
            lengths_of_numbers = [2, 1, 3, 4, 7, 10, 13]
            for number_length in lengths_of_numbers:
                if len(number_as_digit) >= number_length and self.dict_of_ordinal_numbers.get(number_as_digit[-1*number_length:]):
                    ordinal_for_declension = self.dict_of_ordinal_numbers[number_as_digit[-1*number_length:]]
                    break
        
        return ordinal_for_declension


    def __remove_ending_in_number_as_text(self, number_as_text, ending):
        ''' Удалить окончание в числе, записанном словами (числе прописью, например 'двадцать пять-ого').
        
        1. number_as_text - число, записанное словами
        2. ending - окончание (без дефиса)
        3. возвращает обновлённый number_as_text '''

        number_as_text = number_as_text[:len(number_as_text)-len(ending)]
        number_as_text = number_as_text[:-1] if number_as_text[-1] == '-' else number_as_text

        return number_as_text


    def __declension_of_ordinal_number_only(self, words, tags, ordinal_for_declension):
        ''' Просклонять только порядковое число в списке в соответствии с тэгами.
        
        1. words - список исходных слов
        2. tags - тэги для PyMorphy2
        3. ordinal_for_declension - порядковое число для склонения
        4. возвращает обновлённый words с заменённым последним словом на порядковое число в требуемом склонении '''

        updated_word = self.morph_analyzer.parse(ordinal_for_declension)[0].inflect(tags)
        words[-1] = updated_word.word if updated_word else words[-1]

        return words


    def __declension_of_each_word(self, words, tags):
        ''' Просклонять каждое слово в списке в соответствии с тэгами.
        
        1. words - список исходных слов для склонения
        2. tags - тэги для PyMorphy2
        3. возвращает обновлённый words '''

        for i, word in enumerate(words):
            if word and word.isalpha():
                updated_word = self.morph_analyzer.parse(word)[0].inflect(tags)
                words[i] = updated_word.word if updated_word else word
        
        return words


    def decline(self, number_as_text, source_number, replace_with_ordinal=True):
        ''' Просклонять число в соответствии с его окончанием. Окончание может быть записано слитно с числом либо через дефис. Поддерживаются окончания
        для порядковых числительных, которые отвечают за падеж и род, а так же окончание 'летия' (которое сохраняется в исходном числе, например
        '25-летия' -> 'двадцати пяти-летия').

        Примечание: гарантируется корректная работа только с целыми числами до 1000.
        
        1. number_as_text - число, записанное словами (например, 'двадцать один')
        2. source_number - исходное число с окончанием (например, '21-е')
        3. replace_with_ordinal - True: заменять число на порядковое при склонении
        4. возвращает обновлённый number_as_text '''

        if not number_as_text or not source_number:
            return number_as_text

        # Разделение исходного числа на непосредственно число и его окончание
        number_as_digit = ''.join([symbol for symbol in source_number if symbol.isdigit()])
        ending = source_number.replace(number_as_digit, '').replace('-', '')

        if not ending or not number_as_digit:
            return number_as_text

        # Если окончание числа найдено в таблице сопоставления окончаний порядковых чисел и тэгов pymorphy (с падежом и родом)
        if self.dict_of_ordinal_endings_and_pymorphy_tags.get(ending):
            ordinal_for_declension = self.__search_ordinal_number_for_declension(replace_with_ordinal, number_as_digit)
            number_as_text = self.__remove_ending_in_number_as_text(number_as_text, ending)

            tags = self.dict_of_ordinal_endings_and_pymorphy_tags[ending]
            words = self.__tokenize(number_as_text, ignore_hyphen=True)

            # Если было найдено соответствующее порядковое число - склонение только порядкового числа и замена им последнего слова с строке,
            # иначе склонение каждого слова в строке
            if ordinal_for_declension:
                words = self.__declension_of_ordinal_number_only(words, tags, ordinal_for_declension)
            else:
                words = self.__declension_of_each_word(words, tags)
            
            number_as_text = ''.join(words)
        
        # Если окончание числа найдено в таблице сопоставления окончаний обычных чисел и тэгов pymorphy (с падежом и родом)
        elif self.dict_of_stored_endings_and_pymorphy_tags.get(ending):
            tags = self.dict_of_stored_endings_and_pymorphy_tags[ending]
            words = self.__tokenize(number_as_text, ignore_hyphen=True)

            words = self.__declension_of_each_word(words, tags)
            number_as_text = ''.join(words)

        return number_as_text


# Для адресов:
# - опорные слова для смены числа на порядковое: улица, переулок, шоссе, километр

# Добавить поддержку: двухтысячный, ..., одиннадцатитысячный, ..., стотысячный и т.д. (http://old-rozental.ru/orfograf_uk.php?oid=851)

# Статьи про склонение числительных:
# https://russkiiyazyk.ru/chasti-rechi/chislitelnoe/sklonenie-chislitelnyh.html
# https://chislitelnye.ru/sklonenie/
# https://www.yaklass.ru/p/russky-yazik/6-klass/imia-chislitelnoe-10569/prostye-slozhnye-i-sostavnye-chislitelnye-razriady-chislitelnykh-skloneni_-10572/re-866191d9-70f9-4fad-ab2b-17983de1e8bb
# http://www.fio.ru/pravila/grammatika/sklonenie-imen-chislitelnykh/


def main():
    test_texts = {
        ('тридцать два-летия', '32-летия'):
                'тридцати двух-летия',
        ('пятьдесят летия', '50летия'):
                'пятидесяти летия',
        ('тридцать один', '31ого'):
                'тридцать первого',
        ('двадцать пять', '25-ом'):
                'двадцать пятом',
        ('пять', '5-я'):
                'пятая',
        ('два', '2-я'):
                'вторая',
        ('сто', '100-е'):
                'сотое',
        ('тысяча', '1000-й'):
                'тысячный',
        ('тридцать два-летия', '32-летия'):
                'тридцати двух-летия',
        ('пятьдесят летия', '50летия'):
                'пятидесяти летия',
    }

    numerals_declination_ru = NumeralsDeclination_ru()


    print('[i] Тестирование на {} примерах...'.format(len(test_texts)))
    elapsed_times = []
    error_result_tests = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        result = numerals_declination_ru.decline(text[0], text[1])
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


if __name__ == '__main__':
    main()
