#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Восстановление буквы 'ё' в русском тексте, где это необходимо.

Содержит класс YoRestorer. Зависимости отсутствуют.
'''

import time
import re

try:
    from dictionary import YoDictionary
except ModuleNotFoundError:
    from text_preparation.yo_restorer.dictionary import YoDictionary


class YoRestorer:
    ''' Восстановление буквы 'ё' в русском тексте, где это необходимо. Для работы используются 2 вида словарей: с однозначными (safe) и
    спорными (unsafe) словами.

    Весь функционал обеспечивает метод restore().
    
    1. f_name_safe_dict - имя .txt словаря однозначных слов (по умолчанию 'text_preparation/yo_restorer/dicts/safe.txt' или 'dicts/safe.txt')
    2. f_name_unsafe_dict - имя .txt словаря спорных слов (по умолчанию 'text_preparation/yo_restorer/dicts/unsafe.txt' или 'dicts/unsafe.txt')
    3. f_name_add_dict - имя дополнительного .txt словаря (добавляется в словарь однозначных слов) '''

    exception_dict_for_abbreviation = [
        'мед',  # 'мед. училище', 'мед. учреждение' и т.д., сокращение от слова 'медицинский'
    ]

    def __init__(self, f_name_safe_dict=None, f_name_unsafe_dict=None, f_name_add_dict=None):
        self.yo_dict = YoDictionary(f_name_safe_dict, f_name_unsafe_dict, f_name_add_dict)

        self.word_search_re = re.compile(r'([\W\d_])')  # относится к __tokenize и будет удалено

        self.search_word_with_e_re = re.compile(r'\b[а-яё+]*[её]+[а-яё+]*\b', flags=re.IGNORECASE)
        self.replace_e_with_yo_by_index = lambda word, index: word[:index] + ('Ё' if word[index].isupper() else 'ё') + word[index+1:]


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
    

    def __restore_in_word(self, source_word, prepared_word, dict_type='safe', ignore_word_characteristics=False):
        ''' Восстановление буквы 'ё' в одном слове, которое присутствует в словаре.
        
        1. source_word - исходное слово из строки
        2. prepared_word - подготовленное слово (убраны 'ё', '+' и переведено в нижний регистр)
        3. dict_type - 'safe': использовать словарь однозначных слов, 'unsafe': использовать словарь спорных слов
        4. ignore_word_characteristics - True: игнорировать характеристики/признаки слова (употребление только с большой или маленькой буквы)
        5. возвращает source_word с восстановленной 'ё' '''

        yo_word = self.yo_dict[dict_type][prepared_word]
        yo_index = yo_word.with_yo.lower().find('ё')

        while yo_index != -1:
            if yo_word.only_lowercase and not yo_word.only_uppercase and source_word[0].islower():
                source_word = self.replace_e_with_yo_by_index(source_word, yo_index)

            elif not yo_word.only_lowercase and yo_word.only_uppercase and source_word[0].isupper():
                source_word = self.replace_e_with_yo_by_index(source_word, yo_index)
                    
            elif not yo_word.only_lowercase and not yo_word.only_uppercase:
                source_word = self.replace_e_with_yo_by_index(source_word, yo_index)

            elif ignore_word_characteristics:
                source_word = self.replace_e_with_yo_by_index(source_word, yo_index)
            
            next_yo_index = yo_word.with_yo[yo_index+1:].lower().find('ё')
            yo_index = yo_index + next_yo_index + 1 if next_yo_index != -1 else next_yo_index
        
        return source_word


    def __remove_yo_from_abbreviation(self, prepared_word, start_idx, end_idx, text):
        ''' Замена буквы 'ё' на 'е' в сокращениях/аббревиатурах (например, 'мед. учреждение'). Список известных сокращений/аббревиатур
        находится в переменной exception_dict_for_abbreviation.
        
        1. prepared_word - текущее подготовленное слово (убраны 'ё', '+' и переведено в нижний регистр)
        2. start_idx - индекс начала текущего (подготовленного) слова в исходном тексте
        3. end_idx - индекс конца текущего (подготовленного) слова в исходном тексте
        4. text - исходная строка с текстом
        5. возвращает исходный текст с заменёнными 'ё' на 'е' в найденном сокращении/аббревиатуре '''

        updated_text = text
        if prepared_word in self.exception_dict_for_abbreviation and end_idx + 1 < len(text):
            next_punctuation = True if text[end_idx] == '.' else False  # если следующий символ точка

            next_word_in_lowercase = False  # если после точки слово состоит из 2 и более букв и начинается с двух маленьких/больших букв
            i = end_idx + 1
            while i < len(text):
                if i + 1 < len(text) and ((text[i].islower() and text[i+1].islower()) or (text[i].isupper() and text[i+1].isupper())):
                    next_word_in_lowercase = True
                    break
                elif text[i].isspace():
                    i += 1
                else:
                    break

            if next_punctuation and next_word_in_lowercase:
                updated_text = text[:start_idx] + text[start_idx:end_idx].replace('ё', 'е').replace('Ё', 'Е') + text[end_idx:]

        return updated_text


    def restore(self, text, use_unsafe_dict=False, ignore_word_characteristics=False):
        ''' Восстановление буквы 'ё' в тексте, где это необходимо. Разбивает текст на слова и отдельные символы (кроме символа '+' внутри слова),
        ищет совпадения в словарях с 'ё' и если совпадения найдены - заменяет 'е' на 'ё' в исходном тексте.
        
        1. text - текст для восстановления буквы 'ё'
        2. use_unsafe_dict - True: искать совпадения так же и в словаре спорных слов (небезопасном словаре)
        3. ignore_word_characteristics - True: игнорировать характеристики/признаки слова (употребление только с большой или маленькой буквы)
        4. возвращает text с восстановленными 'ё' '''

        # TODO: добавить аргумент enable_warning, при включении которого слова так же будут искаться в unsafe словаре, но вместо восстановления 'ё'
        # в исходном тексте будет возвращаться dict вида {'text': 'исходный текст с ё', 'warning': ['список найденных спорных слов с ё']}

        if not isinstance(text, str) or text.isspace():
            return text

        for match in self.search_word_with_e_re.finditer(text):
            start_idx, end_idx, word = match.start(), match.end(), match.group()

            word = word.replace('ё', 'е').replace('Ё', 'Е')
            prepared_word = word.lower().replace('+', '')

            # Поиск совпадений в словаре однозначных слов
            if self.yo_dict['safe'].get(prepared_word):
                word_with_yo = self.__restore_in_word(word, prepared_word, dict_type='safe', ignore_word_characteristics=ignore_word_characteristics)
                text = text[:start_idx] + word_with_yo + text[end_idx:]

            # Поиск совпадений в словаре спорных слов
            elif use_unsafe_dict and self.yo_dict['unsafe'].get(prepared_word):
                word_with_yo = self.__restore_in_word(word, prepared_word, dict_type='unsafe', ignore_word_characteristics=ignore_word_characteristics)
                text = text[:start_idx] + word_with_yo + text[end_idx:]

            # Удаление 'ё' в сокращениях/аббревиатурах (например, 'мед. учреждение')
            text = self.__remove_yo_from_abbreviation(prepared_word, start_idx, end_idx, text)

        return text


def restorer_tests(test_texts, yo_restorer, **kwargs):
    ''' Тестирование восстановления буквы 'ё' в русском тексте на парах 'исходный текст': 'текст с ё'. Результат тестирования печатается в терминал.
    
    1. test_texts - dict с тестовыми парами формата 'исходный текст': 'текст с ё'
    2. yo_restorer - инициализированный экземпляр класса YoRestorer '''

    elapsed_times = []
    error_result_tests = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        result = yo_restorer.restore(text, **kwargs)
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
    print('\n')


# TODO: при поиске и удалении ё в сокращениях/аббревиатурах ('мёд' и 'мед. учреждение') для определения, нужно удалять ё или нет, используется
# информация о следующем слове (его длина и написано ли оно верхним/нижним регистром). На данный момент алгоритм смотрит только следующие две буквы
# после пробела, нужно переделать на цельное слово! Например, делать срез от текущего слова до конца строки, а затем вычленять следующее слово
# по пробелам

# Примечание: модуль не умеет обрабатывать слова с прилипшими числами ('1ежик' и 'ежик1')! Он их игнорирует, т.к. это уже не его зона ответственности.


def main():
    test_safe_texts = {
        None: None,
        '': '',
        ' ': ' ',
        '   ': '   ',
        12: 12,
        'Корабль': 'Корабль',
        'Ежик': 'Ёжик',
        'елки-палки': 'ёлки-палки',
        'Кухни-дешево.рф': 
                'Кухни-дёшево.рф',
        '12 и 123-89 пять-ноль-пять плю+с':
                '12 и 123-89 пять-ноль-пять плю+с',
        '?12,32+45.0356-0=BOOM-09!':
                '?12,32+45.0356-0=BOOM-09!',
        'киев Киев левою Левою':
                'киёв Киев левою Лёвою',
        'Один ежик и 2 ежика':
                'Один ёжик и 2 ёжика',
        'Ежик\nЕжик\nежик':
                'Ёжик\nЁжик\nёжик',
        'Емко, остроумно и убедительно.':
                'Ёмко, остроумно и убедительно.',
        '«Лед тронулся, господа присяжные заседатели!»':
                '«Лёд тронулся, господа присяжные заседатели!»',
        'Мед. образова+ние иде+т и гряде+т, мед.центр, мед.а он купил мед, мед.Он в мед. УЧРЕЖДЕНИЕ, но не в МЕД. дом':
                'Мед. образова+ние идё+т и грядё+т, мед.центр, мёд.а он купил мёд, мёд.Он в мед. УЧРЕЖДЕНИЕ, но не в МЕД. дом',
        'Бёрёза, еще бёреза, а вот бере+за с ударе+ниями!!':
                'Берёза, ещё берёза, а вот берё+за с ударе+ниями!!',
        'Елочка! Не выкорчевывайся!':
                'Ёлочка! Не выкорчёвывайся!',
        'Еж любит ее и нее ефикатор':
                'Ёж любит её и неё ёфикатор',
        'екарная ее береза, ЕкарнаЯ ЕЕ! БЕРЕЗА? БеРеЗА: нет еще не бЁрЕза':
                'ёкарная её берёза, ЁкарнаЯ ЕЁ! БЕРЁЗА? БеРёЗА: нет ещё не бЕрЁза',
        'слег=слеток(скулеж - сойдет + ссосет)/нет':
                'слёг=слёток(скулёж - сойдёт + ссосёт)/нет',
        'english text не должен чаевничать':
                'english text не должен чаёвничать',
        'АКСЕНОВ ПЕТР АРТЕМОВИЧ':
                'АКСЁНОВ ПЁТР АРТЁМОВИЧ',
        'Аксенов Петр Артемович':
                'Аксёнов Пётр Артёмович',
        'желтощекий мальчик':
                'жёлтощёкий мальчик',
    }

    test_unsafe_texts = {
        'Головенков Клева Очеров - акушер в березневском':
                'Головёнков Клёва Очёров - акушёр в березнёвском',
        'девочка Телегеновых':
                'девочка Тёлёгёновых',
    }

    test_texts_with_unformatted_names = {
        'АКСЕНОВ ПЕТР АРТЕМОВИЧ':
                'АКСЁНОВ ПЁТР АРТЁМОВИЧ',
        'Аксенов Петр Артемович':
                'Аксёнов Пётр Артёмович',
        'аксенов петр артемович':
                'аксёнов пётр артёмович',
        'АкСеНоВ пЕтР аРтЕмОвИч':
                'АкСёНоВ пЁтР аРтЁмОвИч',
        'аКсЕнОв ПеТр АрТеМоВиЧ':
                'аКсЁнОв ПёТр АрТёМоВиЧ',
    }


    f_name_add_dict = 'text_preparation/yo_restorer/additional_yo_dict.txt'
    start_time = time.time()
    yo_restorer = YoRestorer(f_name_add_dict=f_name_add_dict)
    elapsed_time = time.time() - start_time
    print('[i] Время инициализации {:.6f} с или {:.2f} мс\n'.format(elapsed_time, elapsed_time*1000))


    print('[i] Тестирование словаря однозначных слов (безопасного словаря) на {} примерах...'.format(len(test_safe_texts)))
    restorer_tests(test_safe_texts, yo_restorer, use_unsafe_dict=False, ignore_word_characteristics=False)

    print('[i] Тестирование словаря спорных слов (небезопасного словаря) на {} примерах...'.format(len(test_unsafe_texts)))
    restorer_tests(test_unsafe_texts, yo_restorer, use_unsafe_dict=True, ignore_word_characteristics=False)

    print('[i] Тестирование неформатированных ФИО на {} примерах...'.format(len(test_texts_with_unformatted_names)))
    restorer_tests(test_texts_with_unformatted_names, yo_restorer, use_unsafe_dict=True, ignore_word_characteristics=True)


    # Тестирование на вводе пользователя
    while True:
        text = input('\n[i] Введите текст: ')
        start_time = time.time()
        prepared_text = yo_restorer.restore(text, use_unsafe_dict=False, ignore_word_characteristics=False)
        elapsed_time = time.time() - start_time

        print("[i] Результат: '{}'".format(prepared_text))
        print('[i] Время обработки {:.6f} с или {:.2f} мс'.format(elapsed_time, elapsed_time*1000))


if __name__ == '__main__':
    main()
