#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Чтение размеченных слов из xlsx таблиц и конвертация их в json словарь с последующим обновлением уже существующего
в библиотеке stress_dictionary словаря.

Зависимости: pandas==1.0.3 xlrd==1.2.0
'''

import json
import pandas


def xlsx_table_processing(f_name_xlsx_table):
    ''' Чтение xlsx таблицы с 2 столбцами (1 столбец - исходное слово, 2 столбец - слово с поставленным знаком + после гласной ударением),
    её очистка и конвертирование в подходящий формат для библиотеки stress_dictionary.

    1. f_name_xlsx_table - имя .xlsx таблицы
    2. возвращает готовый словарь ударений  '''

    # Чтение xlsx таблицы
    print("[i] Чтение таблицы '{}'...".format(f_name_xlsx_table))
    xlsx_table_df = pandas.read_excel(f_name_xlsx_table, sheet_name=None)
    xlsx_table_df = xlsx_table_df[list(xlsx_table_df)[0]].fillna('')
    print('[i] Загружено {} строк'.format(xlsx_table_df.shape[0]))


    # Удаление строк, в которых нет ударения
    source_row_count = xlsx_table_df.shape[0]
    rows_indexes_to_drop = []
    for i in range(xlsx_table_df.shape[0]):
        if not xlsx_table_df[xlsx_table_df.columns[1]][i]:
            rows_indexes_to_drop.append(i)
    xlsx_table_df = xlsx_table_df.drop(rows_indexes_to_drop)
    print('[i] Удалено {} строк с неразмеченными значениями, осталось {} строк'.format(
            source_row_count-xlsx_table_df.shape[0], xlsx_table_df.shape[0]))


    # Конвертирование таблицы в более удобный словарь {'слово': 'сло+во', ...}
    print("[i] Конвертирование таблицы в словарь формата {'слово': 'сло+во'}...")
    xlsx_table = xlsx_table_df.to_dict()
    xlsx_table_c1 = xlsx_table[xlsx_table_df.columns[0]]
    xlsx_table_c2 = xlsx_table[xlsx_table_df.columns[1]]

    temp_xlsx_table = {}
    for row_1, row_2 in zip(xlsx_table_c1, xlsx_table_c2):
        temp_xlsx_table[xlsx_table_c1[row_1]] = xlsx_table_c2[row_2]
    xlsx_table = temp_xlsx_table


    # Разделение значений, состоящих из нескольких слов, записанных через пробел или дефис, на отдельные значения
    # (т.к. библиотека stress_dictionary пока что не умеет работать с составными значениями в словаре)
    print('[i] Корректировка составных значений...')
    xlsx_table_keys = list(xlsx_table)
    i = 0
    while i < len(xlsx_table_keys):
        if xlsx_table[xlsx_table_keys[i]].find(' ') != -1:
            subwords = xlsx_table[xlsx_table_keys[i]].split(' ')
            for subword in subwords:
                xlsx_table[subword.replace('+', '')] = subword
            del xlsx_table[xlsx_table_keys[i]]
            i += len(subwords) - 1
        elif xlsx_table[xlsx_table_keys[i]].find('-') != -1:
            subwords = xlsx_table[xlsx_table_keys[i]].split('-')
            for subword in subwords:
                xlsx_table[subword.replace('+', '')] = subword
            del xlsx_table[xlsx_table_keys[i]]
            i += len(subwords) - 1
        else:
            i += 1


    # Удаление слов, в которых пропущены ударения
    xlsx_table_keys = list(xlsx_table)
    i = 0
    while i < len(xlsx_table_keys):
        if xlsx_table[xlsx_table_keys[i]].find('+') == -1:
            del xlsx_table[xlsx_table_keys[i]], xlsx_table_keys[i]
        else:
            i += 1


    # Конвертирование словаря в подходящий формат для библиотеки stress_dictionary
    new_stress_dict = {key:[value.find('+')] for key, value in xlsx_table.items()}
    print('[i] В конечном словаре осталось {} значений'.format(len(new_stress_dict)))
    return new_stress_dict




def main():
    f_names_xlsx_tables = ['stressed_names_а.xlsx',
                           'stressed_names_б.xlsx',
                           'stressed_names_в.xlsx',
                           'stressed_names_г_ж.xlsx',
                           'stressed_names_д.xlsx',
                           'stressed_names_е.xlsx',
                           'stressed_names_з.xlsx',
                           'stressed_names_и_й.xlsx',
                           'stressed_names_к.xlsx',
                           'stressed_names_л.xlsx',
                           'stressed_names_м.xlsx',
                           'stressed_names_н.xlsx',
                           'stressed_names_о_п.xlsx',
                           'stressed_names_р.xlsx',
                           'stressed_names_с_т.xlsx',
                           'stressed_names_у.xlsx',
                           'stressed_names_ф.xlsx',
                           'stressed_names_х.xlsx',
                           'stressed_names_ц_ч_ш_щ_ы.xlsx',
                           'stressed_names_э_ю_я.xlsx']
    f_name_new_dictionary = 'update_for_dictionary_names_ru.json'
    f_name_updatable_dictionary = 'text_preparation/stress_dictionary/stress_dictionary/dicts/dictionary_names_ru.json'


    new_stress_dict = {}
    for f_name_xlsx_table in f_names_xlsx_tables:
        temp_dictionary = xlsx_table_processing(f_name_xlsx_table)
        new_stress_dict.update(temp_dictionary)
        print()

    print("[i] Сохранение {} значений в '{}'...".format(len(new_stress_dict), f_name_new_dictionary))
    with open(f_name_new_dictionary, 'w') as f_new_dictionary:
        json.dump(new_stress_dict, f_new_dictionary, sort_keys=True, indent=4, ensure_ascii=False)


    print("[i] Чтение целевого словаря '{}'...".format(f_name_updatable_dictionary))
    with open(f_name_updatable_dictionary, 'r') as f_updatable_dictionary:
        updatable_stress_dict = json.load(f_updatable_dictionary)
    print('[i] Загружено {} значений'.format(len(updatable_stress_dict)))

    source_updatable_stress_dict_len = len(updatable_stress_dict)
    updatable_stress_dict.update(new_stress_dict)
    print('[i] В словарь добавлено {} новых значений'.format(len(updatable_stress_dict)-source_updatable_stress_dict_len))

    print("[i] Сохранение {} значений в '{}'...".format(len(updatable_stress_dict), f_name_updatable_dictionary))
    with open(f_name_updatable_dictionary, 'w') as f_updatable_dictionary:
        json.dump(updatable_stress_dict, f_updatable_dictionary, sort_keys=True, indent=4, ensure_ascii=False)

    print()


if __name__ == '__main__':
    main()
