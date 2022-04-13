#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для загрузки и работы со словарями однозначных и спорных слов с буквой 'ё'.
'''

import os


class YoWord:
    ''' Хранение информации о слове с буквой 'ё'. Содержит поля:

        with_yo - слово с буквой 'ё'
        only_uppercase - True: слово употребляется только при написании с большой буквы (например, как часть ФИО или названия чего-либо)
        only_lowercase - True: слово употребляется только при написании с маленькой буквы (например, "киёв" (от "кий") и "Киев" (город))

    Использование `__slots__` позволяет уменьшить потребление RAM на 30% в сравнении со словарём с аналогичными полями (с 76 Мб до 52 Мб, при
    использовании отладчика в VS Code с 92 Мб до 68 Мб).

    1. word - слово с буквой 'ё' из текстового словаря с удалённым комментарием и раскрытими окончаниями '''

    __slots__ = 'with_yo', 'only_uppercase', 'only_lowercase'

    def __init__(self, word):
        self.only_uppercase = word[0].isupper()
        self.only_lowercase = word[0] == '_'

        self.with_yo = word.lower() if self.only_uppercase else word
        self.with_yo = word[1:] if self.only_lowercase else self.with_yo


class YoDictionary(dict):
    ''' Загрузка и разбор словарей однозначных (safe) и спорных (unsafe) слов с буквой 'ё' из текстовых файлов. Создаёт словарь со структурой:

    {
        'safe': {
            'слово1_без_ё': структура_YoWord,  # с полями with_yo='слово1_с_ё', only_uppercase=True/False, only_lowercase=True/False
            ...
        },
        'unsafe': {
            'слово1_без_ё': структура_YoWord,
            ...
        }
    }
    
    1. f_name_safe_dict - имя .txt словаря однозначных слов (по умолчанию 'text_preparation/yo_restorer/dicts/safe.txt' или 'dicts/safe.txt')
    2. f_name_unsafe_dict - имя .txt словаря спорных слов (по умолчанию 'text_preparation/yo_restorer/dicts/unsafe.txt' или 'dicts/unsafe.txt')
    3. f_name_add_dict - имя дополнительного .txt словаря (добавляется в словарь однозначных слов) '''

    def __init__(self, f_name_safe_dict=None, f_name_unsafe_dict=None, f_name_add_dict=None):
        if not f_name_safe_dict:
            f_name_safe_dict = 'text_preparation/yo_restorer/dicts/safe.txt'
            f_name_safe_dict = 'dicts/safe.txt' if not os.path.exists(f_name_safe_dict) else f_name_safe_dict
        
        if not f_name_unsafe_dict:
            f_name_unsafe_dict = 'text_preparation/yo_restorer/dicts/unsafe.txt'
            f_name_unsafe_dict = 'dicts/unsafe.txt' if not os.path.exists(f_name_unsafe_dict) else f_name_unsafe_dict

        safe_dict = self.__parse_file(f_name_safe_dict)
        unsafe_dict = self.__parse_file(f_name_unsafe_dict)

        if f_name_add_dict:
            safe_dict.update(self.__parse_file(f_name_add_dict))

        super(YoDictionary, self).__init__({'safe': safe_dict, 'unsafe': unsafe_dict})


    def __parse_file(self, f_name_dict):
        ''' Разбор словаря из текстового файла. Читает файл построчно, для каждой строки выполняет удаление комментариев, подстановку окончаний слов из скобок
        и разбор характеристик каждого слова.
        
        1. f_name_dict - имя .txt словаря
        2. возвращает словарь со структурой:
        {
            'слово1_без_ё': структура_YoWord,  # с полями with_yo='слово1_с_ё', only_uppercase=True/False, only_lowercase=True/False
            ...
        } '''

        parsed_dict = {}
        with open(f_name_dict, 'r') as f_dict:
            # TODO: по правильному лучше использовать генераторы, например так:
            # parsed_words = (parsed_word for word in f_dict for parsed_word in self.__parse_source_word(word.replace('\n', '')) if parsed_word)
            # Но это приводит к замедлению инициализации словаря, с 500мс до 1.2с. Нужно разобраться почему и переделать.
            source_word = f_dict.readline().replace('\n', '')
            while source_word:
                parsed_words = self.__parse_source_word(source_word)
                
                source_word = f_dict.readline().replace('\n', '')
                if not parsed_words:
                    continue
                
                # Если вынести создание экземпляра YoWord(word) в метод __parse_source_word(), возрастёт время инициализации словаря на 40-50 мс
                for word in parsed_words:
                    yo_word = YoWord(word)
                    parsed_dict[yo_word.with_yo.replace('ё', 'е')] = yo_word

        return parsed_dict


    def __remove_comment(self, word):
        ''' Удаление комментария после слова. За комментарий принимается любой текст после символа решётки '#'. Если всё слово является комментарием,
        вернёт пустую строку.
        
        1. word - исходное слово из текстового словаря
        2. возвращает обновлённый word '''

        if word.find('#') != -1:
            word = word[:word.find('#')].strip()

        return word


    def __unpack_endings(self, word):
        ''' Раскрытие скобок с окончаниями для слова, если такие есть. Преобразует строку вида 'слов(|а|о)' в список слов вида ['слов', 'слова', 'слово'].
        
        1. word - исходное слово из текстового словаря с удалённым комментарием
        2. возвращает список слов с окончаниями '''

        if word.find('(') != -1:
            stem_word = word[:word.find('(')]
            endings = word[word.find('('):].replace('(', '').replace(')', '').split('|')
            words_with_ending = [stem_word + ending for ending in endings]
        else:
            words_with_ending = [word]
        
        return words_with_ending


    def __parse_source_word(self, source_word):
        ''' Разбор одного слова из текстового словаря. Выполняет удаление комментария и подстановку окончаний слова из скобок.
        
        1. source_word - исходное слово из текстового словаря
        2. возвращает список разобранных слов '''

        word = self.__remove_comment(source_word)

        # Если строка состояла только из комментария
        if not word:
            return word

        words_with_ending = self.__unpack_endings(word)
        return words_with_ending




def main():
    f_name_safe_dict = 'text_preparation/yo_restorer/dicts/safe.txt'
    f_name_unsafe_dict = 'text_preparation/yo_restorer/dicts/unsafe.txt'
    f_name_add_dict = 'text_preparation/yo_restorer/additional_yo_dict.txt'

    yo_dict = YoDictionary(f_name_add_dict=f_name_add_dict)

    pass
    


if __name__ == '__main__':
    main()
