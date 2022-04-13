#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Поиск и расшифровка сокращений/аббревиатур по словарю в русском тексте. Поддерживаются стандартные символы, экранированные с помощью \\ знаки препинания,
базовые простые и составные сокращения, даты, время, языки, числа, деньги и адреса. Список сокращений в каждой категории не претендует на полноту,
но со временем будет увеличиваться.

Для полноты расшифровок модуль должен работать в паре с модулем согласования числительных с рядом стоящими словами.
'''

import re
import time


# Словари из пар "сокращение - расшифровка", заточенный под нашу ПрО
# Символы
ABBREVIATIONS_SYMBOLS = {
    '#': 'решётка',
    '%': 'процент',
    '^': 'карет',
    '&': 'и',
    '*': 'звёздочка',
    '/': 'слэш',
    '=': 'равно',
    '@': 'собака',
    '\\\\': 'обратный слэш',
    '_': 'нижнее подчёркивание',
    '|': 'или',
    '~': 'примерно',
    '№': 'номер',
}

# Экранированные знаки препинания
ABBREVIATIONS_ESCAPED_PUNCTUATION = {    
    '\\,': 'запятая',
    '\\.': 'точка',
    '\\:': 'двоеточие',
    '\\;': 'точка с запятой',
    '\\?': 'вопросительный знак',
    '\\!': 'восклицательный знак',
    '\\(': 'открывающаяся скобка',
    '\\)': 'закрывающаяся скобка',
    '\\_': 'нижнее подчёркивание',
    '\\-': 'тирэ+',
    '\\+': 'плюс',
    '\\\'': 'кавычка',
}

# Базовые простые сокращения
ABBREVIATIONS_SIMPLE_BASIC = {
    'адр': 'адрес',
    'смс': 'эс эм эс',
    'cл': 'следующий',
    'тел': 'телефон',
    'гос-во': 'государство',
    'гос': 'государственный',
    'гр-н': 'гражданин',
    'гр-не': 'граждане',
    'гражд': 'гражданский',
    'ден': 'денежный',
    'др': 'другое',
    'ежедн.': 'ежедневный',
    'ежемес': 'ежемесячный',
    'еженед': 'еженедельный',
    'ж-д': 'железнодорожный',
    'жен': 'женский',
    #'ж': 'женский',  # указано ниже
    'муж': 'мужской',
    #'м': 'мужской',  # указано ниже
    'изм': 'изменение',
    'изобр': 'изображение',
    'иностр': 'иностранный',
    'инст': 'институт',
    'инф': 'информация',
    'информ': 'информация',
    'лк': 'личный кабинет',
    'лс': 'личные сообщения',
    #'макс': 'максимальный',  # может совпасть с именем Макс, имена приоритетнее
    #'мин': 'минимальный',  # приоритетнее 'мин' -> 'минута' (пока что несколько вариантов расшифровки не поддерживаются)
    'междунар': 'международный',
    'междн': 'международный',
    'напр': 'например',
    'табл': 'таблица',
    'им': 'имя',
    'фам': 'фамилия',
    'отч': 'отчество',
    'пр': 'прочее',
    'чел': 'человек',
    'яз': 'язык',
    'р-р': 'раствор',
    'прим': 'примечание',
}

# Базовые составные сокращения (из нескольких слов) (несколько вариантов одного и того же сокращения пишутся вручную для упрощения
# алгоритма поиска совпадений)
ABBREVIATIONS_COMPLEX_BASIC = {
    'м. б': 'может быть',
    'м.б': 'может быть',
    'м б': 'может быть',
    'мб': 'может быть',
    'м/б': 'может быть',

    'т. д': 'так далее',
    'т.д': 'так далее',
    'т д': 'так далее',
    'тд': 'так далее',

    'т. п': 'тому подобное',
    'т.п': 'тому подобное',
    'т п': 'тому подобное',
    'тп': 'тому подобное',

    'т. е': 'то есть',
    'т.е': 'то есть',
    'т е': 'то есть',
    #'те': 'то есть',  # может совпасть с предлогом 'те' ('те люди', 'вон те дома')

    'ф. и. о': 'фамилия имя отчество',
    'ф. и о': 'фамилия имя отчество',
    'ф и. о': 'фамилия имя отчество',
    'ф и о': 'фамилия имя отчество',
    'ф.и.о': 'фамилия имя отчество',
    'ф.ио': 'фамилия имя отчество',
    'фи.о': 'фамилия имя отчество',
    'фио': 'фамилия имя отчество',
}

# Дни недели
ABBREVIATIONS_DAYS_OF_WEEK = {
    'пн': 'понедельник',
    'пнд': 'понедельник',
    'вт': 'вторник',
    'втр': 'вторник',
    'ср': 'среда',
    'срд': 'среда',
    'чт': 'четверг',
    'чтв': 'четверг',
    'пт': 'пятница',
    'птн': 'пятница',
    'сб': 'суббота',
    'сбт': 'суббота',
    'вс': 'воскресенье',
    'вскр': 'воскресенье',
}

# Месяцы
ABBREVIATIONS_MONTHS = {
    'янв': 'январь',
    'фев': 'февраль',
    'мар': 'март',
    'апр': 'апрель',
    'май': 'май',
    'июн': 'июнь',
    'июл': 'июль',
    'авг': 'август',
    'сен': 'сентябрь',
    'окт': 'октябрь',
    'ноя': 'ноябрь',
    'дек': 'декабрь',
}

# Время
ABBREVIATIONS_TIME = {    
    'нс': 'наносекунда',
    'мкс': 'микросекунда',
    'мс': 'миллисекунда',
    'сек': 'секунда',
    'мин': 'минута',
    'ч': 'час',
    'сут': 'сутки',
    #'д': 'день',  # расшифровка 'д' -> 'дом' приоритетнее
    'дн': 'дней',
    'нд': 'неделя',
    'нед': 'неделя',
    #'м': 'месяц',  # расшифровка 'м' -> 'мужской' приоритетнее, вариант 'м' -> 'месяц' очень редкий
    'мес': 'месяц',
    #'г': 'год',  # указано ниже
}

# Расстояния
ABBREVIATIONS_DISTANCES = {
    'км': 'километр',
    'м.': 'метр',
    #'см.': 'сантиметр',  # может совпасть с сокращением 'см.' -> 'смотрите'
    'мм.': 'миллиметр',
}

# Языки согласно ISO 639: https://ru.wikipedia.org/wiki/Коды_языков
ABBREVIATIONS_LANGUAGES = {
    'рус': 'русский',
    'бел': 'беларусский',
    'укр': 'украинский',
    'анг': 'английский',
    'нем': 'немецкий',
    'фра': 'французский',
}

# Числа
ABBREVIATIONS_NUMBERS = {
    'тыс': 'тысяча',
    'млн': 'миллион',
    'млрд': 'миллиард',
    'трлн': 'триллион',
}

# Деньги
ABBREVIATIONS_MONEY = {
    'бел. руб': 'белорусский рубль',
    'бел.руб': 'белорусский рубль',
    'бел руб': 'белорусский рубль',

    'бел. р': 'белорусский рубль',
    'бел.р': 'белорусский рубль',
    'бел р': 'белорусский рубль',

    'б. руб': 'белорусский рубль',
    'б.руб': 'белорусский рубль',
    'б руб': 'белорусский рубль',

    'б. р': 'белорусский рубль',
    'б.р': 'белорусский рубль',
    'б р': 'белорусский рубль',

    'р': 'рубль',
    'руб': 'рубль',

    'рос. руб': 'российский рубль',
    'рос.руб': 'российский рубль',
    'рос руб': 'российский рубль',

    'рос. р': 'российский рубль',
    'рос.р': 'российский рубль',
    'рос р': 'российский рубль',

    'р. руб': 'российский рубль',
    'р.руб': 'российский рубль',
    'р руб': 'российский рубль',

    'р. р': 'российский рубль',
    'р.р': 'российский рубль',
    'р р': 'российский рубль',

    'дол': 'доллар',
    'долл': 'доллар',
    'евр': 'евро',
    '$': 'доллар',
    '€': 'евро',
    '₽': 'рубль',
}

# Адреса, источник: https://www.alta.ru/fias/socrname/
ABBREVIATIONS_ADDRESSES = {
    #'рф': 'Российская Федерация',
    #'рб': 'Республика Беларусь',
    'спб': 'Санкт-Петербург',
    'мск': 'Москва',
    'екб': 'Екатеринбург',
    'нск': 'Новосибирск',

    #'г': 'город',  # указано ниже
    'с.п': 'сельское поселение',
    'сп': 'сельское поселение',
    'с.': 'село',
    #'д': 'деревня',  # указано ниже
    'м-ко': 'местечко',
    'х.': 'хутор',

    'кп': 'курортный посёлок',
    'г.п': 'городской посёлок',
    'гп': 'городской посёлок',
    #'п': 'посёлок',  # указано ниже
    'пос': 'посёлок',
    'о.п': 'остановочный пункт',
    'пос. о.п': 'посёлок остановочного пункта',
    'пос о.п': 'посёлок остановочного пункта',
    'пос.о.п': 'посёлок остановочного пункта',
    'рп': 'рабочий посёлок',
    'дп': 'дачный посёлок',
    'пгт': 'посёлок городского типа',
    'нп': 'населённый пункт',
    'снт': 'садоводческое некоммерческое товарищество',

    'обл': 'область',
    'а.обл.': 'автономная область',
    'окр': 'округ',
    'а.окр.': 'автономный округ',
    'ф.окр.': 'федеральный округ',
    'ф.о': 'федеральный округ',
    'г.о': 'городской округ',
    'респ': 'республика',

    'ж.д': 'железная дорога',
    'жд': 'железная дорога',
    'ж/д': 'железная дорога',
    'ст': 'станция',
    'ж/д рзд': 'железнодорожный разъезд',
    'ж/д ст.': 'железнодорожная станция',
    'пос. ж/д ст.': 'посёлок при железнодорожной станции',
    'пос ж/д ст.': 'посёлок при железнодорожной станции',
    'п. ж/д ст.': 'посёлок при железнодорожной станции',
    'ж/д о.п': 'железнодорожный остановочный пункт',
    'ж/д к-т': 'железнодорожный комбинат',
    'ж/д пл-ма': 'железнодорожная платформа',
    'ж/д пл-ка': 'железнодорожная площадка',

    'р-н': 'район',
    'м.р-н': 'муниципальный район',
    'вн.р-н': 'внутригородской район',
    'ж/р': 'жилой район',
    'кв-л': 'квартал',
    'мкр': 'микрорайон',
    'п/р': 'промышленный район',

    'с/с': 'сельсовет',
    'г.ф.з.': 'город федерального значения',
    'тер.': 'территория',

    'б-р': 'бульвар',
    'взд': 'въезд',
    'дор.': 'дорога',
    'ззд': 'заезд',
    'к-цо': 'кольцо',
    'лн.': 'линия',
    'мгстр': 'магистраль',
    'наб.': 'набережная',
    'пер-д': 'переезд',
    'пер.': 'переулок',
    'пл': 'площадь',
    'пр-зд': 'проезд',
    'пр-д': 'проезд',
    'пр-кт': 'проспект',
    'пр-т': 'проспект',
    'рзд': 'разъезд',
    'с-р': 'сквер',
    'с-к': 'спуск',
    'сзд': 'съезд',
    'туп.': 'тупик',
    'ул': 'улица',
    'ш.': 'шоссе',

    'влд.': 'владение',
    'г-ж': 'гараж',
    #'д.': 'дом',  # указано ниже
    'двлд': 'домовладение',
    'зд': 'здание',
    'з/у': 'земельный участок',
    'кв': 'квартира',
    'ком.': 'комната',
    'подв.': 'подвал',
    'п-б': 'погреб',
    #'к.': 'корпус',  # указано ниже
    'корп.': 'корпус',
    'пав': 'павильон',
    'пом': 'помещение',
    'помещ': 'помещение',
    'раб.уч': 'рабочий участок',
    'скл': 'склад',
    'coop': 'сооружение',
    'стр': 'строение',
    'торг.зал': 'торговый зал',
    'уч.': 'учреждение',
    'я/с': 'ясли-сад',
}

ABBREVIATIONS_OTHER = {}

ABBREVIATIONS_DAYS_OF_WEEK = {
    **{key_1 + '-' + key_2: ABBREVIATIONS_DAYS_OF_WEEK[key_1] + '-' + ABBREVIATIONS_DAYS_OF_WEEK[key_2]
            for key_1 in ABBREVIATIONS_DAYS_OF_WEEK for key_2 in ABBREVIATIONS_DAYS_OF_WEEK},
    **ABBREVIATIONS_DAYS_OF_WEEK
}

ABBREVIATIONS = {
    **ABBREVIATIONS_SYMBOLS,
    **ABBREVIATIONS_ESCAPED_PUNCTUATION,
    **ABBREVIATIONS_SIMPLE_BASIC,
    **ABBREVIATIONS_COMPLEX_BASIC,
    **ABBREVIATIONS_DAYS_OF_WEEK,
    **ABBREVIATIONS_MONTHS,
    **ABBREVIATIONS_TIME,
    **ABBREVIATIONS_DISTANCES,
    **ABBREVIATIONS_LANGUAGES,
    **ABBREVIATIONS_NUMBERS,
    **ABBREVIATIONS_ADDRESSES,
    **ABBREVIATIONS_OTHER
}

# Сокращения, которые бывают только перед числами
ABBREVIATIONS_BEFORE_NUMBERS = {
    'ст': 'строение',
    'стр': 'строение',
    'д': 'дом',
    'к': 'корпус',
    'оф': 'офис',
}

# Сокращения, которые бывают только после чисел
ABBREVIATIONS_AFTER_NUMBERS = {
    'г': 'год',
    **ABBREVIATIONS_MONEY,
}

# Сокращения, рядом с которыми не может быть цифр
ABBREVIATIONS_WITHOUT_NUMBERS_NEAR = {
    'ж': 'женский',
    'м': 'мужской',
    'д': 'деревня',
    'п': 'посёлок',
    'г': 'город',
}

# Сокращения, которые записываются только большими буквами
ABBREVIATIONS_ONLY_BIG_LETTERS = {
    'КЛХ': 'колхоз',
    'СВХ': 'совхоз',
}

# Сложные в произношении сокращения
ABBREVIATIONS_WITH_DIFFICULT_PRONUNCIATION = {
    'АС': 'ассоциация',
    'АО': 'акционерное общество',
    'ОО': 'общественная организация',
    'ОАО': 'открытое акционерное общество',
    'ЗАО': 'закрытое акционерное общество',
    'ООО': 'общество с ограниченной ответственностью',
}


class ExpandingAbbreviations_ru:
    ''' Поиск и расшифровка сокращений/аббревиатур по словарю в русском тексте. Поддерживаются стандартные символы, экранированные с помощью \\ знаки
    препинания, базовые простые и составные сокращения, даты, время, языки, числа, деньги и адреса. '''

    def __init__(self):
        self.word_search_re = re.compile(r'([\W\d_])')
        self.search_multiple_spaces = re.compile(r'\s{2,}')
        self.search_for_initials_in_names_re = re.compile(r'(^|[^А-ЯЁа-яё])[А-ЯЁ](\.|\s|\.\s)[А-ЯЁ]($|[^А-ЯЁа-яё])')
        self.search_for_track_names_re = re.compile(r'([АЕМНР]-?\d+)')

        self.is_extended_alnum = lambda word: word.replace('+', '').isalnum()
        self.max_number_words_in_abbreviations = max([len(self.__tokenize(abbreviation)) for abbreviation in ABBREVIATIONS])
    

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


    def __search_letters_at_beginning_and_end_window(self, current_window):
        ''' Поиск в текущем окне начальных и конечных пробелов и цифр.
        
        1. current_window - текущее окно с объединёнными в строку словами
        2. возвращает tuple из строк с начальными и конечными пробелами и цифрами '''

        not_letters_at_beginning = ''
        not_letters_at_end = ''
        
        for symbol in current_window:
            if symbol.isdigit() or symbol.isspace():
                not_letters_at_beginning += symbol
            else:
                break
                
        for symbol in reversed(current_window):
            if symbol.isdigit() or symbol.isspace():
                not_letters_at_end += symbol
            else:
                break
        not_letters_at_end = ''.join(reversed(not_letters_at_end))

        return not_letters_at_beginning, not_letters_at_end


    def __add_space_before_abbreviation(self, not_letters_at_beginning, words, i):
        ''' Добавление пробела перед сокращением, если это необходимо.
        
        1. not_letters_at_beginning - строка с начальными пробелами и цифрами
        2. words - исходный список слов (токенов)
        3. i - индекс первого слова текущего окна в исходном списке слов
        4. возвращает обновлённый not_letters_at_beginning '''

        # Добавление пробела перед сокращением в случае, когда в начальных символах отсутствует пробел
        if not_letters_at_beginning and not not_letters_at_beginning[-1].isspace():
            not_letters_at_beginning += ' '

        # Исключение случаев, когда пробел перед сокращением добавлять не нужно:
        # - в начальных символах присутствует пробел
        # - предыдущий токен пробел или не буква/цифра
        # - первый токен в строке
        if not_letters_at_beginning and not_letters_at_beginning[-1].isspace():
            add_space_before_abbreviation = False
        elif i - 1 > 0 and (words[i-1].isspace() or not self.is_extended_alnum(words[i-1])):
            add_space_before_abbreviation = False
        elif i == 0:
            add_space_before_abbreviation = False
        else:
            add_space_before_abbreviation = True
                
        if add_space_before_abbreviation:
            not_letters_at_beginning += ' '
        
        return not_letters_at_beginning


    def __add_space_after_abbreviation(self, not_letters_at_end, current_window, words, i, window_len):
        ''' Добавление пробела после сокращения, если это необходимо.
        
        1. not_letters_at_end - строка с конечными пробелами и цифрами
        2. current_window - текущее окно с объединёнными в строку словами
        3. words - исходный список слов (токенов)
        4. i - индекс первого слова текущего окна в исходном списке слов
        5. window_len - количество слов в окне
        6. возвращает обновлённый not_letters_at_end '''

        # Добавление пробела после сокращения в случае, когда в конечных символах отсутствует пробел или сокращение заканчивается точкой
        if (not_letters_at_end and not not_letters_at_end[0].isspace()) \
                or (not_letters_at_end and not not_letters_at_end[0].isspace() and current_window and current_window[-1] == '.'):
            not_letters_at_end = ' ' + not_letters_at_end

        # Исключение случаев, когда пробел после сокращения добавлять не нужно:
        # - в конечных символах присутствует пробел
        # - следующий токен пробел или не буква/цифра
        # - последний токен в строке
        if not_letters_at_end and not_letters_at_end[0].isspace():
            add_space_after_abbreviation = False
        elif i + window_len < len(words) and (words[i+window_len].isspace() or not self.is_extended_alnum(words[i+window_len])):
            add_space_after_abbreviation = False
        elif i + window_len == len(words):
            add_space_after_abbreviation = False
        else:
            add_space_after_abbreviation = True

        if add_space_after_abbreviation:
            not_letters_at_end = ' ' + not_letters_at_end
        
        return not_letters_at_end
    

    def __expand_abbreviation_in_window(self, not_letters_at_beginning, not_letters_at_end, expanded_abbreviation):
        ''' Расшифровка скоращения/аббревиатуры в текущем окне. '''
        
        current_window = not_letters_at_beginning + expanded_abbreviation + not_letters_at_end
        return self.__tokenize(current_window)


    def expand(self, text, expand_difficult_abbreviations=True):
        ''' Поиск и расшифровка сокращений/аббревиатур по словарю в русском тексте. Поддерживаются стандартные символы, экранированные с помощью \\ знаки
        препинания, базовые простые и составные сокращения, даты, время, языки, числа, деньги и адреса.

        Для полноты расшифровок модуль должен работать в паре с модулем согласования числительных с рядом стоящими словами.

        Примечение: в случае последовательности 'сокращение.(слово|Слово|число)' точка будет заменена пробелом, в случае 'сокращение. (слово|Слово|число)' -
        точка будет удалена.

        1. text - строка с текстом для обработки
        2. expand_difficult_abbreviations - True: расшифровывать сложные в произношении сокращения
        3. возвращает обработанный текст '''

        if not text or text.isspace():
            return text

        # Удаление нескольких подряд идущих пробелов (использование text.strip() немного ломает перевод текста с фонемами
        # в последовательность чисел и обратно)
        text = self.search_multiple_spaces.sub(' ', text)

        # Разбиение строки на слова и отдельные символы, и объединение слов, у которых указано ударение или которые записаны через -
        words = self.__tokenize(text)

        # Исправление особенности работы токенизатора: объединение экранирующего символа \ и следующую за ним не букву
        i = 0
        while i < len(words) - 1:
            if words[i] == '\\' and not words[i+1].isalpha():
                words[i] += words[i+1]
                del words[i+1]
            else:
                i += 1

        # Объединение сокращений, записанных через /
        i = 1
        while i < len(words) - 1:
            if words[i] == '/' and words[i-1] != ' ' and words[i+1] != ' ' and ABBREVIATIONS.get(words[i-1]+words[i]+words[i+1]):
                words[i-1] = words[i-1]+words[i]+words[i+1]
                del words[i:i+2]
            else:
                i += 1

        # Основной цикл поиска и расшифровки сокращений
        # Алгоритм работы: окном размера max_number_words_in_abbreviations выполняется перебор слов. На каждой итерации во вложенном цикле,
        # ищутся совпадения с окном в словаре сокращений. Если совпадение найдено - расшифровка сокращения и переход к следующему слову. Если совпадение
        # не найдено - размер окна уменьшается на 1 слово и поиск повторяется.
        i = 0
        while i < len(words):
            window_len = self.max_number_words_in_abbreviations
            if i + window_len > len(words):
                window_len = len(words) - i

            while window_len > 0 and i + window_len <= len(words):
                current_window = ''.join(words[i:i+window_len])
                if current_window.isspace():
                    break

                # Исключение из расшифровки:
                # - сокращений инициалов в ФИО вида 'И. О.', 'И.О.' и 'И О', например 'Клим Г. О.'
                # - названий трасс из 1 заглавной буквы (А, Е, М, Н, Р) и следующим за ней числом (как слитно, так и через дефис) вида 'М12', 'М-12'
                if self.search_for_initials_in_names_re.search(current_window) or self.search_for_track_names_re.search(current_window):
                    i += window_len
                    break

                # Поиск в окне начальных и конечных пробелов и цифр
                not_letters_at_beginning, not_letters_at_end = self.__search_letters_at_beginning_and_end_window(current_window)

                # Поиск цифр в отделённых фрагментах
                is_number_at_beginning = True if not_letters_at_beginning.replace(' ', '').isdigit() else False
                is_number_at_end = True if not_letters_at_end.replace(' ', '').isdigit() else False 

                # Удаление в окне начальных и конечных пробелов и цифр
                current_window = current_window[len(not_letters_at_beginning):len(current_window)-len(not_letters_at_end)]

                # Если символ экранирования был отделён - возвращение его обратно к сокращению
                if current_window and not_letters_at_end and current_window[-1] == '\\':
                    current_window += not_letters_at_end[0]
                    not_letters_at_end = not_letters_at_end[1:]
                

                # Добавление пробела перед сокращением, если это необходимо
                not_letters_at_beginning = self.__add_space_before_abbreviation(not_letters_at_beginning, words, i)

                # Добавление пробела после сокращения, если это необходимо
                not_letters_at_end = self.__add_space_after_abbreviation(not_letters_at_end, current_window, words, i, window_len)


                current_window_l = current_window.lower()
                current_window_lp = current_window_l[:-1] if current_window_l and current_window_l[-1] == '.' else current_window_l
                current_window_p = current_window[:-1] if current_window and current_window[-1] == '.' else current_window

                # Расшифровка сокращений, которые бывают только перед числами и пишутся только с точкой
                if ABBREVIATIONS_BEFORE_NUMBERS.get(current_window_l) and is_number_at_end:
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_BEFORE_NUMBERS[current_window_l])
                    break

                # Расшифровка сокращений, которые бывают только перед числами
                elif ABBREVIATIONS_BEFORE_NUMBERS.get(current_window_lp) and is_number_at_end:
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_BEFORE_NUMBERS[current_window_lp])
                    break

                # Расшифровка сокращений, которые бывают только после чисел и пишутся только с точкой
                elif ABBREVIATIONS_AFTER_NUMBERS.get(current_window_l) and is_number_at_beginning:
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_AFTER_NUMBERS[current_window_l])
                    break
                
                # Расшифровка сокращений, которые бывают только после чисел
                elif ABBREVIATIONS_AFTER_NUMBERS.get(current_window_lp) and is_number_at_beginning:
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_AFTER_NUMBERS[current_window_lp])
                    break

                # Расшифровка сокращений, которые записываются только большими буквами и пишутся только с точкой
                elif ABBREVIATIONS_ONLY_BIG_LETTERS.get(current_window):
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_ONLY_BIG_LETTERS[current_window])
                    break
                
                # Расшифровка сокращений, которые записываются только большими буквами
                elif ABBREVIATIONS_ONLY_BIG_LETTERS.get(current_window_p):
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_ONLY_BIG_LETTERS[current_window_p])
                    break

                # Расшифровка сложных в произношении сокращений, которые пишутся только с точкой
                elif ABBREVIATIONS_WITH_DIFFICULT_PRONUNCIATION.get(current_window):
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_WITH_DIFFICULT_PRONUNCIATION[current_window])
                    break

                # Расшифровка сложных в произношении сокращений
                elif ABBREVIATIONS_WITH_DIFFICULT_PRONUNCIATION.get(current_window_p):
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_WITH_DIFFICULT_PRONUNCIATION[current_window_p])
                    break

                # Расшифровка сокращений, рядом с которыми не может быть цифр и которые пишутся только с точкой
                elif ABBREVIATIONS_WITHOUT_NUMBERS_NEAR.get(current_window_l) and not is_number_at_beginning and not is_number_at_end:
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_WITHOUT_NUMBERS_NEAR[current_window_l])
                    break

                # Расшифровка сокращений, рядом с которыми не может быть цифр
                elif ABBREVIATIONS_WITHOUT_NUMBERS_NEAR.get(current_window_lp) and not is_number_at_beginning and not is_number_at_end:
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS_WITHOUT_NUMBERS_NEAR[current_window_lp])
                    break

                # Пропуск сокращений, рядом с которыми не может быть цифр или дефиса, но при этом цифра или дефис найдены
                elif ABBREVIATIONS_WITHOUT_NUMBERS_NEAR.get(current_window_l.replace('-', '')) and (is_number_at_beginning or is_number_at_end \
                        or current_window_l[0] == '-' or current_window_l[-1] == '-'):
                    i += window_len
                    break

                # Расшифровка всех остальных сокращений, которые пишутся только с точкой
                elif ABBREVIATIONS.get(current_window_l):
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS[current_window_l])
                    break

                # Расшифровка всех остальных сокращений
                elif ABBREVIATIONS.get(current_window_lp):
                    words[i:i+window_len] = self.__expand_abbreviation_in_window(not_letters_at_beginning, not_letters_at_end,
                                                                                 ABBREVIATIONS[current_window_lp])
                    break

                window_len -= 1
            i += 1

        text = ''.join(words)
        return text


# Замечания:
# 1. Что бы сделать полноценную правильную расшифровку сокращений нужно учитывать контекст, хотя бы передавать его аргументом,
# например context='time'/'date'/'address' и т.д.
# 2. Учитывать в алгоритме поиска совпадений возможное наличие точек и пробелов в составных сокращениях, что бы не прописывать все варианты вручную
# в словаре сокращений

# Разные списки сокращений:
# http://new.gramota.ru/spravka/docs?layout=item&id=16_15 (согласно ГОСТ Р 7.0.12-2011)
# https://popravilam.com/blog/sokrashcheniya-osnovnye.html

# TODO: сделать полный рефакторинг словаря! Вместо разделения на типы сокращений добавить параметры/характеристики каждому сокращению. Например:
#   'г': {
#       'город': {
#           'только_перед_числом': False,
#           'только_после_числа': False,
#           'только_без_чисел': True,
#           'только_большими_буквами': False,
#           'только_маленькими_буквами': False,
#           'только_с_точкой': False
#       },
#       'год': {
#           'только_перед_числом': False,
#           'только_после_числа': True,
#           'только_без_чисел': False,
#           'только_большими_буквами': False,
#           'только_маленькими_буквами': False,
#           'только_с_точкой': False
#       }
#   }


def main():
    test_texts = {
        '': '',
        '  ': '  ',
        '^тел. Привет ф. и. о.за 5$, пн, сб/вс /ирод диро/ и пр!*?Дава+й№сво+й тел., напиши мне м/б.\\\'] смс.[\\\\ \\.=, и /, а потом пошли в ж-д ' + \
        'инст.по адр ф.о. Колумбия, г Екб, пр-т.3Ленина, д.9, пом.18, оф12. Ну и т.д., и тп.':
                'карет телефон Привет фамилия имя отчество за 5 доллар, понедельник, суббота слэш воскресенье слэш ирод диро слэш и прочее!' + \
                'звёздочка?Дава+й номер сво+й телефон, напиши мне может быть кавычка] эс эм эс[обратный слэш точка равно, и слэш, а потом пошли в ' + \
                'железнодорожный институт по адрес федеральный округ Колумбия, город Екатеринбург, проспект 3Ленина, дом 9, помещение 18, офис 12. Ну ' + \
                'и так далее, и тому подобное',
        '24,5\\+65=\\?&\\-\\+=\\(':
                '24,5 плюс 65 равно вопросительный знак и тирэ+ плюс равно открывающаяся скобка',
        'Привет чел.! Дай свой №тел. и пр пожалуйста. Позвоню в пн-вт, по адр екб, пл.Чёрная, д5 корп.2, пом. 11, оф.9. И м/б куплю чего\\-нить. Всё, \\.':
                'Привет человек! Дай свой номер телефон и прочее пожалуйста. Позвоню в понедельник-вторник, по адрес Екатеринбург, площадь Чёрная, дом 5 ' + \
                'корпус 2, помещение 11, офис 9. И может быть куплю чего тирэ+ нить. Всё, точка',
        'Головин Т.Д., Клим Г О, Клим ГО, Головин Т. Д., Ярцевич С. Х.':
                'Головин Т.Д., Клим Г О, Клим ГО, Головин Т. Д., Ярцевич С. Х.',
        '2-3 случая на (5-10) мин и 23 000 000 000 р 6':
                '2-3 случая на (5-10) минута и 23 000 000 000 рубль 6',
        'дом 34, ст34, стр65, ст.34, стр.2, ст 4, стр 8, ст Новая, ст.Старая, стр. пять':
                'дом 34, строение 34, строение 65, строение 34, строение 2, строение 4, строение 8, станция Новая, станция Старая, строение пять',
        'вот это ас! АС 30 лет победы, ООО Ракушка (но не ооо рак)':
                'вот это ас! ассоциация 30 лет победы, общество с ограниченной ответственностью Ракушка (но не ооо рак)',
        'ул. 25км, трасса Р52 53км, трасса М2 43 км':
                'улица 25 километр, трасса Р52 53 километр, трасса М2 43 километр',
        'Трассы М1 М-2, Р-153, Р89, дома 89а, Б23, корпус Г1, строение 94т, подвал32-в он шёл с 2 сумками':
                'Трассы М1 М-2, Р-153, Р89, дома 89а, Б23, корпус Г1, строение 94т, подвал32-в он шёл с 2 сумками',
        'дом 54б, стр39 литера9 павильон 2В':
                'дом 54б, строение 39 литера9 павильон 2В',
        'ул. Ангарская, уч. 5, ст.2, уч.Администрации, ул.мельника':
                'улица Ангарская, учреждение 5, строение 2, учреждение Администрации, улица мельника',
        '25р р и затем в д90 05р д.Выселки д.8 2021г г.Минск':
                '25 российский рубль и затем в дом 90 05 рубль деревня Выселки дом 8 2021 год город Минск',
        'М12, М-12 М0, А03 А-23 Р5, Р-9, Е1, Е-123, Н20, Н-123':
                'М12, М-12 М0, А03 А-23 Р5, Р-9, Е1, Е-123, Н20, Н-123',
        'дом 5Д, корпус 1Г, помещение 17М строение 8Ж, этаж 4П дом 5д, 5-Д, 5-д, корпус 1г, 1-Г, 1-г, помещение 17м, 17-М, 17-м':
                'дом 5Д, корпус 1 год, помещение 17М строение 8Ж, этаж 4П дом 5д, 5-Д, 5-д, корпус 1 год, 1-Г, 1-г, помещение 17м, 17-М, 17-м',
        'Д5, корпус Г1, помещение М17 строение Ж8, этаж П4, д5 Д-5, д-5, корпус г1, Г-1 г-1, помещение м17 М-17 м-17':
                'дом 5, корпус Г1, помещение М17 строение Ж8, этаж П4, дом 5 Д-5, д-5, корпус г1, Г-1 г-1, помещение м17 М-17 м-17',
        'Привет мир! Как твои дела? Сколько тебе лет? Как твоё самочувствие? Случайная фраза на случайную тему':
                'Привет мир! Как твои дела? Сколько тебе лет? Как твоё самочувствие? Случайная фраза на случайную тему',
    }

    expanding_abbreviations_ru = ExpandingAbbreviations_ru()

    print('[i] Тестирование на {} примерах...'.format(len(test_texts)))
    elapsed_times = []
    error_result_tests = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        result = expanding_abbreviations_ru.expand(text)
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


    print('\n[i] Тестирование на вводе пользователя')
    while True:
        text = input('\n[i] Введите текст: ')
        start_time = time.time()
        result = expanding_abbreviations_ru.expand(text)
        elapsed_time = time.time() - start_time

        print("[i] Результат: '{}'".format(result))
        print('[i] Время обработки {:.6f} с или {:.2f} мс'.format(elapsed_time, elapsed_time*1000))


if __name__ == '__main__':
    main()
