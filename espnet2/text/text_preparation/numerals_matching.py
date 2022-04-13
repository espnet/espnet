#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Kazakov A., Klim V. O.

import re
from typing import List, Union, Tuple
from pymorphy2 import MorphAnalyzer


class NumeralsMatching_ru(object):
    ''' Класс для согласования слов с числительными. Склоняет переданный список слов и число с помощью PyMorphy2. '''

    # Список исключений, которые некорректно определяются словарем в pymorphy2
    # `годов/лет` являются дубликатами - http://opencorpora.org/dict.php?act=edit&id=72149
    dictionary_exceptions = {
        'годов': 'лет'
    }

    # Слова, которые нужно пропускать при согласовании
    skipped_words = [
        'лет'
    ]

    # Слова, при обнаружении которых согласование пропускается
    stop_words = [
        'дом',
        'строение',
        'здание',
        'корпус',
        'помещение',
        'офис',
        'квартира',
        'комната',
        'подвал',
        'гараж',
        'литера',
        'павильон'
    ]

    punctuation_marks = [',', '.', '!', '?', ':', ';', '(', ')']

    number_inflectable = [1, 2]
    number_exceptions = [11, 12]

    supported_preposition = ['с', 'со', 'до']

    def __init__(self):
        self.main_word = None
        self.morph_analyzer = MorphAnalyzer()


    def make_agree(self, number: Union[int, float], words: List[str], number_as_text: str = None) -> List[str]:
        ''' Согласовать слова с числительным. Например, `один бутылка вина` -> `одна бутылка вина`, `два красные бутылка` -> `две красные бутылки`.
    
        1. number - число, с которым требуется согласовать слова
        2. words - слова, идущие после числа, которые требуется согласовать
        3. number_as_text - число, с которым требуется согласовать слова, записанное словами (если необходимо поставить само число в правильную форму)
        4. returns обновлённый words или list из обновлённых [number_as_text, *words] '''

        # Если в строковом представлении числа обнаружены стоп-символы (знаки препинания)
        if number_as_text and (number_as_text[-1] in self.punctuation_marks or number_as_text[0] in ['('] \
                               or number_as_text.rfind('-') != -1):
            return [number_as_text, *words]

        # Если нет слов для согласования
        if not words or (len(words) == 1 and words[0].isspace()):
            return [number_as_text]

        # Если следующее слово после числа содержит цифры
        digits_in_first_word = [symbol for symbol in words[0] if symbol.isdigit()]
        if digits_in_first_word:
            return [number_as_text, *words]
        

        # PyMorphy2 правильно работает только с положительными числами
        if number < 0:
            number *= -1

        words, words_after_stop_word = self.__get_words_after_stop_word(words)
        words, words_after_punctuation_mark = self.__get_words_after_punctuation_mark(words)
        words, extra_puncts = self.__separate_punctuation_marks_from_each_word(words)

        # TODO: перенести сюда согласование числа с предлогом и если предлог был найден - не выполнять согласование со словами, а только с предлогом
        # TODO: pymorphy2.parse() съедает заглавные буквы, нужно реализовать их восстановление
        parsed_words = [self.morph_analyzer.parse(word)[0] for word in words]


        # Основной цикл согласования слов с числом
        for i, word in enumerate(parsed_words):
            if not word or (word and word.word in self.skipped_words):
                continue

            # Если находим существительное - останавливаем итерацию, считая его конечной точкой
            # Существительное перед склонением приводится к словарной форме/нормализируется (из-за проблемы со словом `гривна`)
            if word.tag.POS == 'NOUN':
                self.main_word = word
                agreed_word = word.normalized.make_agree_with_number(number)
                if agreed_word:
                    words[i] = self.__check_dictionary_exception(agreed_word.word)
                break
            else:
                self.main_word = None

            agreed_word = word.make_agree_with_number(number)
            if agreed_word:
                words[i] = self.__check_dictionary_exception(agreed_word.word)


        words = self.__restore_punctuation_marks_in_each_word(words, extra_puncts)
        words += words_after_punctuation_mark
        words += words_after_stop_word

        if number_as_text:
            number_as_text = self.__inflect_gender_by_subject(number, number_as_text)
            return [number_as_text, *words]
        else:
            return words


    def __get_words_after_stop_word(self, words: List[str]) -> Tuple[List[str], List[str]]:
        ''' Получить все слова после первого найденного стоп слова, если оно есть.
        
        1. words - исходный список слов
        2. возвращает tuple из списка слов перед стоп словом и списка слов после стоп слова (вместе со стоп словом) '''

        words_after_stop_word = []
        for i, word in enumerate(words):
            if word in self.stop_words:
                words_after_stop_word = words[i:]
                words = words[:i]
                break
        
        return words, words_after_stop_word


    def __get_words_after_punctuation_mark(self, words: List[str]) -> Tuple[List[str], List[str]]:
        ''' Получить все слова после первого найденного знака препинания, если он есть.
        
        1. words - исходный список слов
        2. возвращает tuple из списка слов перед знаком препинания (вместе со знаком препинания) и списка слов после знака препинания  '''

        words_after_punctuation_mark = []
        for i, word in enumerate(words):
            found_punctuation_mark = False
            for symbol in word:
                if symbol in self.punctuation_marks:
                    found_punctuation_mark = True
                    break
            if found_punctuation_mark:
                words_after_punctuation_mark = words[i+1:]
                words = words[:i+1]
                break
        
        return words, words_after_punctuation_mark


    def __separate_punctuation_marks_from_each_word(self, words: List[str]) -> Tuple[List[str], List[List[str]]]:
        ''' Отделить знаки препинания от каждого слова, если они есть. Необходимо из-за того, что PyMorphy2 некорректно обрабатывает слова
        вместе с прилипшими к ним знаками препинания.
        
        1. words - исходный список слов
        2. возвращает tuple из списка слов без знаков препинания и списка из отделённых знаков препинания перед и после каждого слова '''

        extra_puncts = [['', ''] for i in range(len(words))]
        for i, word in enumerate(words):
            for symbol in word:
                if not symbol.isalpha():
                    extra_puncts[i][0] += symbol
                else:
                    break
            word = word[len(extra_puncts[i][0]):]

            for symbol in word[::-1]:
                if not symbol.isalpha():
                    extra_puncts[i][1] += symbol
                else:
                    break
            extra_puncts[i][1] = ''.join([symbol for symbol in reversed(extra_puncts[i][1])])
            words[i] = word[:len(word)-len(extra_puncts[i][1])]
        
        return words, extra_puncts


    def __restore_punctuation_marks_in_each_word(self, words: List[str], extra_puncts: List[List[str]]) -> List[str]:
        ''' Восстановить знаки препинания в каждом слове, если они были.
        
        1. words - список слов без знаков препинания
        2. extra_puncts - список из отделённых знаков препинания перед и после каждого слова
        3. возвращает words с восстановленными знаками препинания '''

        words = [extra_puncts[i][0] + word + extra_puncts[i][1] for i, word in enumerate(words)]
        return words


    def __check_dictionary_exception(self, word: str) -> str:
        return self.dictionary_exceptions.get(word) or word


    def __inflect_gender_by_subject(self, number: Union[int, float], number_as_text: str) -> str:
        is_last_digit_in_inflectable = (number % 10) in self.number_inflectable and not (number % 100) in self.number_exceptions

        has_subject = (self.main_word is not None)
        if not number_as_text or not has_subject or not is_last_digit_in_inflectable:
            return number_as_text

        tokens = number_as_text.split(' ')
        tokens[len(tokens)-1] = self.morph_analyzer.parse(tokens[len(tokens)-1])[0].inflect({self.main_word.tag.gender}).word
        return ' '.join(tokens)


    def make_agree_with_preposition(self, preposition: str, number_as_text: str) -> str:
        ''' Согласовать число с предлогом.

        Поддерживаются следующие предлоги:
            'с', 'со', 'до' - число приводится к родительному падежу, например 'с девять' => 'с девяти', 'до сто семьдесят два' => 'до ста семьдесят двух'

        1. preposition - слово перед числом
        2. number_as_text - число, записанное словами (например, 'двадцать один')
        3. returns обновлённый number_as_text '''
        
        # TODO: добавить изменение предлога в зависимости от следующего слова (например, 'с сто' => 'со ста') и корректировки к склонению (например,
        # слово 'сто' неправильно приводятся к Р.п.)

        if preposition.strip().lower() in self.supported_preposition:
            words = number_as_text.split(' ')
            for i, word in enumerate(words):
                if word not in ['ноль']:
                    parsed_word = self.morph_analyzer.parse(word)[0]
                    words[i] = parsed_word.inflect({'gent'}).word
            number_as_text = ' '.join(words)
        return number_as_text


# Доделать:
# 1. Добавить изменение предлога в зависимости от следующего слова (например, 'с сто' => 'со ста')
# 2. Добавить корректировки к склонению (например, слово 'сто' неправильно приводятся к Р.п.)
