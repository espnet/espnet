#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для создания начальных словарей однозначных и спорных слов с буквой 'ё' из словарей от проектов:
1. https://github.com/e2yo/eyo-kernel
2. https://github.com/link2xt/yoficator
3. https://github.com/kalashnikovisme/karamzin
'''

import yaml


def load_dict_from_yoficator(f_name_dict):
    ''' Загрузка и разбор словаря от проекта https://github.com/link2xt/yoficator
    Слова, начинающиеся с "* " являются спорными: наличие ё в них зависит от контекста и употребления.
    
    1. f_name_dict - имя .txt словаря
    2. возвращает tuple из списка однозначных слов с буквой ё и списка спорных слов с буквой ё '''

    print("[i] Загрузка и разбор словаря '{}' от проекта https://github.com/link2xt/yoficator".format(f_name_dict))
    with open(f_name_dict, 'r') as f_dict:
        source_dict = f_dict.read().split('\n')
    
    safe_dict = []
    unsafe_dict = []
    for word in source_dict:
        if not word.strip():
            continue

        if word.find('* ') != -1:
            unsafe_dict.append(word.replace('* ', ''))
        else:
            safe_dict.append(word)
    
    return safe_dict, unsafe_dict


def load_dict_from_karamzin(f_name_dict):
    ''' Загрузка и разбор словаря от проекта https://github.com/kalashnikovisme/karamzin
    
    Примечание: словарь не содержит информации о спорных словах!
    
    1. f_name_dict - имя .yml словаря
    2. возвращает список слов с буквой ё '''

    print("[i] Загрузка и разбор словаря '{}' от проекта https://github.com/kalashnikovisme/karamzin".format(f_name_dict))
    with open(f_name_dict, 'r') as f_dict:
        source_dict = yaml.safe_load(f_dict)['words']
    
    for i, word in enumerate(source_dict):
        word, yo_index = word.split(' ')
        yo_index = int(yo_index)
        word = word[:yo_index] + 'ё' + word[yo_index+1:]
        source_dict[i] = word

    return source_dict


def __parse_dict_from_eyo_kernel(source_dict):
    ''' Разбор словаря от проекта https://github.com/e2yo/eyo-kernel
    Выполняет удаление комментариев и подстановку окончаний слов из скобок.
    
    1. source_dict - список исходных слов из словаря
    2. возвращает полный (разобранный) список слов из словаря '''

    full_dict = []
    for word in source_dict:
        # Удаление комментария
        if word.find('#') != -1:
            word = word[:word.find('#')].strip()

        # Если строка состояла только из комментария
        if not word:
            continue

        # Раскрытие скобок с окончаниями слова
        if word.find('(') != -1:
            stem_word = word[:word.find('(')]#.lower()
            endings = word[word.find('('):].replace('(', '').replace(')', '').split('|')
            full_dict += [stem_word + ending for ending in endings]
        else:
            full_dict.append(word)
    
    return full_dict


def load_dict_from_eyo_kernel(f_name_safe_dict, f_name_unsafe_dict):
    ''' Загрузка и разбор словарей от проекта https://github.com/e2yo/eyo-kernel
    Проект содержит 2 разных словаря: словарь с однозначными словами и со спорными словами.
    
    1. f_name_safe_dict - имя .txt словаря однозначных слов
    2. f_name_unsafe_dict - имя .txt словаря спорных слов
    3. возвращает tuple из списка однозначных слов с буквой ё и списка спорных слов с буквой ё '''

    print("[i] Загрузка и разбор словарей '{}' и '{}' от проекта https://github.com/e2yo/eyo-kernel".format(f_name_safe_dict, f_name_unsafe_dict))
    with open(f_name_safe_dict, 'r') as f_dict:
        safe_dict = f_dict.read().split('\n')

    with open(f_name_unsafe_dict, 'r') as f_dict:
        unsafe_dict = f_dict.read().split('\n')

    safe_dict = __parse_dict_from_eyo_kernel(safe_dict)
    unsafe_dict = __parse_dict_from_eyo_kernel(unsafe_dict)

    return safe_dict, unsafe_dict


def create_unsafe_dict(eyo_kernel_safe_dict, eyo_kernel_unsafe_dict, yoficator_unsafe_dict):
    ''' Создание конечного словаря спорных слов из словарей спорных слов от eyo-kernel и yoficator.
    
    Алгоритм работы:
        1. Словарь спорных слов от eyo-kernel принимается в качестве исходного
        2. Удаление разметки из словарей от eyo-kernel
        3. Поиск совпадений между словарями спорных слов от yoficator и eyo-kernel
        4. Удаление найденных совпадений из словаря спорных слов от yoficator
        5. Поиск совпадений между словарём однозначных слов от eyo-kernel и оставшихся спорных слов от yoficator
        6. Удаление найденных совпадений из словаря спорных слов от yoficator
        7. Добавление оставшихся значений в словарь спорных слов от eyo-kernel
    
    1. eyo_kernel_safe_dict - список однозначных слов от проекта eyo-kernel
    2. eyo_kernel_unsafe_dict - список спорных слов от проекта eyo-kernel
    3. yoficator_unsafe_dict - список спорных слов от проекта yoficator
    4. возвращает обновлённый eyo_kernel_unsafe_dict '''
    
    print('[i] Исходный словарь спорных слов содержит {} слов(-о,-а), обновление словаря...'.format(len(eyo_kernel_unsafe_dict)))

    # Удаление разметки из словарей от eyo-kernel
    eyo_kernel_safe_dict_simple = [word.lower().replace('_', '') for word in eyo_kernel_safe_dict]
    eyo_kernel_unsafe_dict_simple = [word.lower().replace('_', '') for word in eyo_kernel_unsafe_dict]

    # Поиск совпадений между словарями спорных слов от yoficator и eyo-kernel
    yoficator_and_eyo_kernel_unsafe_matches = []
    for word in yoficator_unsafe_dict:
        if word in eyo_kernel_unsafe_dict_simple:
            yoficator_and_eyo_kernel_unsafe_matches.append(word)
    
    # Удаление найденных совпадений из словаря спорных слов от yoficator
    for unsafe_match in yoficator_and_eyo_kernel_unsafe_matches:
        yoficator_unsafe_dict.remove(unsafe_match)

    # Поиск совпадений между словарём однозначных слов от eyo-kernel и оставшихся спорных слов от yoficator
    yoficator_and_eyo_kernel_safe_matches = []
    for word in yoficator_unsafe_dict:
        if word in eyo_kernel_safe_dict_simple:
            yoficator_and_eyo_kernel_safe_matches.append(word)

    # Удаление найденных совпадений из словаря спорных слов от yoficator
    for safe_match in yoficator_and_eyo_kernel_safe_matches:
        yoficator_unsafe_dict.remove(safe_match)

    # Добавление оставшихся значений в словарь спорных слов от eyo-kernel
    eyo_kernel_unsafe_dict = sorted(list(set(eyo_kernel_unsafe_dict+yoficator_unsafe_dict)))
    print('[i] Конечный словарь спорных слов содержит {} слов(-о,-а)'.format(len(eyo_kernel_unsafe_dict)))

    return eyo_kernel_unsafe_dict


def create_safe_dict(eyo_kernel_safe_dict, eyo_kernel_unsafe_dict, yoficator_safe_dict, karamzin_dict):
    ''' Создание конечного словаря однозначных слов из словарей однозначных слов от eyo-kernel, yoficator и karamzin.
    
    Алгоритм работы:
        1. Словарь однозначных слов от eyo-kernel принимается в качестве исходного
        2. Удаление разметки из словарей от eyo-kernel
        3. Поиск совпадений между словарём однозначных слов от yoficator и словарём спорных слов от eyo-kernel
        4. Удаление найденных совпадений из словаря однозначных слов от yoficator
        5. Поиск совпадений между словарём однозначных слов от eyo-kernel и оставшихся однозначных слов от yoficator
        6. Удаление найденных совпадений из словаря однозначных слов от yoficator
        7. Удаление неправильных значений из словаря однозначных слов от yoficator, найденных вручную
        8. Добавление оставшихся значений в словарь однозначных слов от eyo-kernel
        9. Поиск совпадений между словарём от karamzin и словарём спорных слов от eyo-kernel
        10. Удаление найденных совпадений из словаря от karamzin
        11. Поиск совпадений между словарём однозначных слов от eyo-kernel и оставшихся слов от karamzin
        12. Удаление найденных совпадений из словаря от karamzin
        13. Удаление неправильных значений из словаря от karamzin, найденных вручную
        14. Добавление оставшихся значений в словарь однозначных слов от eyo-kernel
    
    1. eyo_kernel_safe_dict - список однозначных слов от проекта eyo-kernel
    2. eyo_kernel_unsafe_dict - список спорных слов от проекта eyo-kernel
    3. yoficator_safe_dict - список однозначных слов от проекта yoficator
    4. karamzin_dict - список всех слов от проекта karamzin
    5. возвращает обновлённый eyo_kernel_safe_dict '''

    print('\n[i] Исходный словарь однозначных слов содержит {} слов(-о,-а), обновление словаря...'.format(len(eyo_kernel_safe_dict)))

    # Удаление разметки из словарей от eyo-kernel
    eyo_kernel_safe_dict_simple = [word.lower().replace('_', '') for word in eyo_kernel_safe_dict]
    eyo_kernel_unsafe_dict_simple = [word.lower().replace('_', '') for word in eyo_kernel_unsafe_dict]


    # Поиск совпадений между словарём однозначных слов от yoficator и словарём спорных слов от eyo-kernel
    print('[i] Поиск и удаление совпадений между словарём однозначных слов от yoficator и словарём спорных слов от eyo-kernel')
    yoficator_and_eyo_kernel_unsafe_matches = []
    for word in yoficator_safe_dict:
        if word in eyo_kernel_unsafe_dict_simple:
            yoficator_and_eyo_kernel_unsafe_matches.append(word)

    # Удаление найденных совпадений из словаря однозначных слов от yoficator
    for unsafe_match in yoficator_and_eyo_kernel_unsafe_matches:
        yoficator_safe_dict.remove(unsafe_match)

    # Поиск совпадений между словарём однозначных слов от eyo-kernel и оставшихся однозначных слов от yoficator
    print('[i] Поиск и удаление совпадений между словарём однозначных слов от eyo-kernel и оставшихся однозначных слов от yoficator')
    yoficator_and_eyo_kernel_safe_matches = []
    for word in yoficator_safe_dict:
        if word in eyo_kernel_safe_dict_simple:
            yoficator_and_eyo_kernel_safe_matches.append(word)

    # Удаление найденных совпадений из словаря однозначных слов от yoficator
    for safe_match in yoficator_and_eyo_kernel_safe_matches:
        yoficator_safe_dict.remove(safe_match)

    # Удаление неправильных значений из словаря однозначных слов от yoficator, найденных вручную
    errors_in_yoficator_safe_dict = ['всплёсне', 'вёсельна', 'жёлтозём', 'новоизобрётенн', 'полуведёрн']
    yoficator_safe_dict_correct = []
    for word in yoficator_safe_dict:
        is_error_word = False
        for error_word in errors_in_yoficator_safe_dict:
            if word.find(error_word) != -1:
                is_error_word = True
        if not is_error_word:
            yoficator_safe_dict_correct.append(word)
    yoficator_safe_dict = yoficator_safe_dict_correct

    # Добавление оставшихся значений в словарь однозначных слов от eyo-kernel
    eyo_kernel_safe_dict = sorted(list(set(eyo_kernel_safe_dict+yoficator_safe_dict)))
    eyo_kernel_safe_dict_simple = [word.lower().replace('_', '') for word in eyo_kernel_safe_dict]
    print('[i] Обновлённый словарь однозначных слов содержит {} слов(-о,-а)'.format(len(eyo_kernel_safe_dict)))


    # Поиск совпадений между словарём от karamzin и словарём спорных слов от eyo-kernel
    print('[i] Поиск и удаление совпадений между словарём от karamzin и словарём спорных слов от eyo-kernel')
    karamzin_and_eyo_kernel_unsafe_matches = []
    for word in karamzin_dict:
        if word in eyo_kernel_unsafe_dict_simple:
            karamzin_and_eyo_kernel_unsafe_matches.append(word)

    # Удаление найденных совпадений из словаря от karamzin
    for unsafe_match in karamzin_and_eyo_kernel_unsafe_matches:
        karamzin_dict.remove(unsafe_match)

    # Поиск совпадений между словарём однозначных слов от eyo-kernel и оставшихся слов от karamzin
    print('[i] Поиск и удаление совпадений между словарём однозначных слов от eyo-kernel и оставшихся слов от karamzin')
    karamzin_and_eyo_kernel_safe_matches = []
    for word in karamzin_dict:
        if word in eyo_kernel_safe_dict_simple:
            karamzin_and_eyo_kernel_safe_matches.append(word)

    # Удаление найденных совпадений из словаря от karamzin
    for safe_match in karamzin_and_eyo_kernel_safe_matches:
        karamzin_dict.remove(safe_match)

    # Удаление неправильных значений из словаря от karamzin, найденных вручную
    errors_in_karamzin_dict = ['вёреш', 'вручённы', 'жёлтощек', 'жёсткозакрепленн', 'перекрёстноопыленн', 'тёлегеновск', 'трёхведерн', 'трёхверст',
                               'трёхзвезд', 'трёхименн', 'трёхколесн', 'трёхпролетн', 'трёхрублев', 'трёхсотрублев', 'трёхшерстн', 'четырёхведерн',
                               'четырёхверстн', 'четырёхвесельн', 'четырёхзвездочн', 'четырёхколесн', 'четырёхпролетн', 'шёстрем']
    karamzin_dict_correct = []
    for word in karamzin_dict:
        is_error_word = False
        for error_word in errors_in_karamzin_dict:
            if word.find(error_word) != -1:
                is_error_word = True
        if not is_error_word:
            karamzin_dict_correct.append(word)
    karamzin_dict = karamzin_dict_correct

    # Добавление оставшихся значений в словарь однозначных слов от eyo-kernel
    eyo_kernel_safe_dict = sorted(list(set(eyo_kernel_safe_dict+karamzin_dict)))
    print('[i] Конечный словарь однозначных слов содержит {} слов(-о,-а)'.format(len(eyo_kernel_safe_dict)))

    return eyo_kernel_safe_dict


def __update_dict_from_eyo_kernel(source_dict, final_dict):
    ''' Добавление новых значений в исходный словарь от eyo-kernel с сохранением его форматирования. Выполняет удаление комментариев, подстановку
    окончаний слов из скобок, поиск и удаление совпадений в финальном словаре, а затем добавление в исходный словарь оставшихся значений из финального
    словаря.

    1. source_dict - список исходных слов из словаря
    2. final_dict - новый список исходных слов для словаря
    3. возвращает обновлённый source_dict '''

    source_dict.remove('')

    for word in source_dict:
        # Удаление комментария
        if word.find('#') != -1:
            word = word[:word.find('#')].strip()

        # Если строка состояла только из комментария
        if not word:
            continue

        # Раскрытие скобок с окончаниями слова, поиск и удаление совпадений в финальном словаре
        if word.find('(') != -1:
            stem_word = word[:word.find('(')]
            endings = word[word.find('('):].replace('(', '').replace(')', '').split('|')
            word_forms = [stem_word + ending for ending in endings]

            for word_form in word_forms:
                if word_form in final_dict:
                    final_dict.remove(word_form)
        else:
            if word in final_dict:
                final_dict.remove(word)
    updated_final_dict = sorted(list(set(source_dict+final_dict)))

    return updated_final_dict


def save_final_dict(eyo_kernel_safe_dict, eyo_kernel_unsafe_dict, f_name_eyo_kernel_safe_dict, f_name_eyo_kernel_unsafe_dict,
                    f_name_final_safe_dict, f_name_final_unsafe_dict):
    ''' Сохранение конечных словарей однозначных и спорных слов. Загружает исходные словари от проекта eyo-kernel, добавляет в них новые значения
    с сохранением исходного форматирования и сохраняет полученные словари.
    
    1. eyo_kernel_safe_dict - обновлённый (конечный) список однозначных слов от eyo-kernel
    2. eyo_kernel_unsafe_dict - обновлённый (конечный) список спорных слов от eyo-kernel
    3. f_name_eyo_kernel_safe_dict - имя .txt словаря с однозначными словами от eyo-kernel
    4. f_name_eyo_kernel_unsafe_dict - имя .txt словаря со спорными словами от eyo-kernel
    5. f_name_final_safe_dict - имя .txt файла для сохранения конечного словаря однозначных слов
    6. f_name_final_unsafe_dict - имя .txt файла для сохранения конечного словаря спорных слов '''

    print("\n[i] Сохранение конечных словарей в '{}' и '{}'".format(f_name_final_safe_dict, f_name_final_unsafe_dict))

    with open(f_name_eyo_kernel_safe_dict, 'r') as f_dict:
        safe_dict = f_dict.read().split('\n')

    with open(f_name_eyo_kernel_unsafe_dict, 'r') as f_dict:
        unsafe_dict = f_dict.read().split('\n')

    updated_safe_dict = __update_dict_from_eyo_kernel(safe_dict, eyo_kernel_safe_dict)
    updated_unsafe_dict = __update_dict_from_eyo_kernel(unsafe_dict, eyo_kernel_unsafe_dict)

    with open(f_name_final_safe_dict, 'w') as f_dict:
        f_dict.writelines([word + '\n' for word in updated_safe_dict])
    
    with open(f_name_final_unsafe_dict, 'w') as f_dict:
        f_dict.writelines([word + '\n' for word in updated_unsafe_dict])


# В конечном словаре исправить вручную:
# 1. Перенести из unsafe в safe:
#     "трёхвёдерн(ая|ого|ое|ой|ом|ому|ую|ые|ый|ым|ыми|ых)"
#     "четырёхвёдерн(ая|ого|ое|ой|ом|ому|ую|ые|ый|ым|ыми|ых)"
#     "четырёхвёсельн(ая|ого|ое|ой|ом|ому|ую|ые|ый|ым|ыми|ых)"
#     "Аксёнов(|а|е|ой|у|ы|ым|ыми|ых)"
#     "Аксёнчик(|а|ам|ами|ах|е|и|ов|ом|у)"
#     "Пётр"
#     "Яхменёв(|а|е|ой|у|ы|ым|ыми|ых)"
#     "Ячменёв(|а|е|ой|у|ы|ым|ыми|ых)"
# 2. Перенести из unsafe в safe и исправить:
#     "жёлтощёк(|а|ая|и|ие|ий|им|ими|их|о|ого|ое|ой|ом|ому|ою|ую)" -> "желтощёк(|а|ая|и|ие|ий|им|ими|их|о|ого|ое|ой|ом|ому|ою|ую)"
# 3. Добавить в safe:
#     "трёхядерн(ая|ого|ое|ой|ом|ому|ую|ые|ый|ым|ыми|ых)"
#     "четырёхядерн(ая|ого|ое|ой|ом|ому|ую|ые|ый|ым|ыми|ых)"
#     "выкорчёвывайтесь"


def main():
    f_name_yoficator_dict = 'text_preparation/yo_restorer/source_dicts/yo.txt'
    yoficator_safe_dict, yoficator_unsafe_dict = load_dict_from_yoficator(f_name_yoficator_dict)

    f_name_karamzin_dict = 'text_preparation/yo_restorer/source_dicts/dictionary.yml'
    karamzin_dict = load_dict_from_karamzin(f_name_karamzin_dict)

    f_name_eyo_kernel_safe_dict = 'text_preparation/yo_restorer/source_dicts/safe.txt'
    f_name_eyo_kernel_unsafe_dict = 'text_preparation/yo_restorer/source_dicts/not_safe.txt'
    eyo_kernel_safe_dict, eyo_kernel_unsafe_dict = load_dict_from_eyo_kernel(f_name_eyo_kernel_safe_dict, f_name_eyo_kernel_unsafe_dict)


    # Словари от проекта eyo-kernel имеют больший приоритет, т.к. этот проект неоднократно рекомендовали в профильных чатах и его словари
    # имеют дополнительную разметку
    print('\n[i] Словари от проекта eyo-kernel приняты как исходные')


    # Создание конечного словаря спорных слов из словарей спорных слов от eyo-kernel и yoficator
    eyo_kernel_unsafe_dict = create_unsafe_dict(eyo_kernel_safe_dict, eyo_kernel_unsafe_dict, yoficator_unsafe_dict)

    # Создание конечного словаря однозначных слов из словарей однозначных слов от eyo-kernel, yoficator и karamzin
    eyo_kernel_safe_dict = create_safe_dict(eyo_kernel_safe_dict, eyo_kernel_unsafe_dict, yoficator_safe_dict, karamzin_dict)


    f_name_final_safe_dict = 'text_preparation/yo_restorer/dicts/safe.txt'
    f_name_final_unsafe_dict = 'text_preparation/yo_restorer/dicts/unsafe.txt'
    save_final_dict(eyo_kernel_safe_dict, eyo_kernel_unsafe_dict, f_name_eyo_kernel_safe_dict, f_name_eyo_kernel_unsafe_dict,
                    f_name_final_safe_dict, f_name_final_unsafe_dict)


if __name__ == '__main__':
    main()
