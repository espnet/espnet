#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Поиск и транскрипция аббревиатур, инициалов в ФИО и буквенно-цифровых обозначений в русском тексте. Транскрипция выполняется в соответствии с
транскрипцией/проищношением букв в русском алфавите.
'''

import re
import time

try:
    from text_preparation.framing_numbers.framing_numbers import NumberWithSpacesCase
    from text_preparation.numerals_declination import NumeralsDeclination_ru
except ModuleNotFoundError:
    from framing_numbers.framing_numbers import NumberWithSpacesCase
    from numerals_declination import NumeralsDeclination_ru


class WordsToAlphabeticalTranscription_ru:
    ''' Поиск и транскрипция аббревиатур, инициалов в ФИО и буквенно-цифровых обозначений в русском тексте. Транскрипция выполняется в соответствии с
    транскрипцией/проищношением букв в русском алфавите. '''

    # Известные транскрипции аббревиатур
    transcriptions = {
        'B2B': 'би ту би',
        'B2C': 'би ту си',
        'ИП': 'и пэ',
        'ГУ': 'гэ у',

        'АС': 'а ээс',
        'АО': 'а о',
        'ОО': 'о о',
        'ОАО': 'о а о',
        'ЗАО': 'ЗАО',
        'ООО': 'о о о',

        'СНО': 'ээс эн о',
        'ОНО': 'о-эн о',
        'ДНО': 'дэ эн о',
        'СНТ': 'ээс эн тэ',
        'ОНТ': 'о-эн тэ',
        'ДНТ': 'дэ эн тэ',
        'СПК': 'эс пэ ка',
        'ОПК': 'о-пэ ка',
        'ДПК': 'дэ пэ ка',
        'СНП': 'ээс эн пэ',
        'ОНП': 'о-эн пэ',
        'ДНП': 'дэ эн пэ',
        'ТСН': 'тэ ээс эн',
        'ГСК': 'гэ ээс ка',
        'ОНС': 'о-эн ээс',

        'ГУП': 'ГУП',
        'ФКП': 'эф ка пэ',
        'МУУП': 'МУУП',
        'ДУП': 'дэ у пэ',
        'ДП': 'дэ пэ',
        'ПК': 'пэ ка',
        'СХК': 'ээс хэ ка',
        'ПТ': 'пэ тэ',
        'ТВ': 'тэ вэ',
        'ОДО': 'ОДО',
        'ФЛ': 'эф эл',
        'ПРЕД': 'ПРЕД',
        'РО': 'эр о',
        'ООБ': 'о о бэ',
        'РОБ': 'РОБ',
        'НП': 'эн пэ',
        'ГУЧ': 'ГУЧ',
        'МУЧ': 'МУЧ',
        'ОУЧ': 'ОУЧ',
        'АНО': 'АНО',
        'ТСЖ': 'тэ ээс жэ',
        'ПТК': 'пэ тэ ка',
        'ОД': 'о-дэ',
        'ОФ': 'о-эф',
        'ООС': 'ООС',
        'ДОУч': 'ДОУч',
        'ППО': 'пэ пэ о',
        'АОЗТ': 'АОЗ тэ',
        'АООТ': 'АОО тэ',
        'ТОО': 'ТОО',
        'МП': 'эм пэ',
        'ИЧП': 'ИЧП',
        'СЕМ': 'СЕМ',
        'КФХ': 'кэ эф ха',
        'КХ': 'кэ ха',
        'СП': 'ээс пэ',
        'ГП': 'гэ пэ',
        'МУП': 'МУП',
        'ПОО': 'пэ о-о',
        'ППКООП': 'пэ пэ ка о-о пэ',
        'УОО': 'УОО',
        'УЧПТК': 'уч пэ тэ ка',
        'СМТ': 'ээс эм тэ',
        'СТ': 'ээс тэ',
        'ЖСК': 'жэ ээс ка',
        'ГСК': 'гэ ээс ка',
        'ЖКХ': 'жэ кэ ха',
        'ЖЭС': 'ЖЭС',
        'НПО': 'эн пэ о',
        'ПО': 'пэ о',
        'СКБ': 'эс ка бэ',
        'КБ': 'ка бэ',
        'УПТК': 'у пэ тэ ка',
        'СМУ': 'эс эм у',
        'ХОЗУ': 'ХОЗУ',
        'НТЦ': 'эн тэ цэ',
        'ФИК': 'ФИК',
        'НПП': 'эн пэ пэ',
        'ЧИФ': 'ЧИФ',
        'ЧОП': 'ЧОП',
        'РЭУ': 'РЭУ',
        'ПИФ': 'ПИФ',
        'ГКООП': 'гэ ка о-о пэ',
        'ПОБ': 'ПОБ',
        'ПС': 'пэ эс',
        'ФФ': 'эф эф',
        'ФПГ': 'эф пэ гэ',
        'МХП': 'эм ха пэ',
        'ЛПХ': 'эл пэ ха',
        'АП': 'а пэ',
        'ОП': 'о пэ',
        'НПФ': 'эн пэ эф',
        'ПКФ': 'пэ ка эф',
        'ПКП': 'пэ ка пэ',
        'ПКК': 'пэ ка ка',
        'КФ': 'ка эф',
        'ТФ': 'тэ эф',
        'ТД': 'тэ дэ',
        'Д(С)У': 'дэ эс у',
        'ТФПГ': 'тэ эф пэ гэ',
        'МФПГ': 'эм эф пэ гэ',
        'Д/С': 'дэ эс',
        'АДОК': 'а дэ ОК',
        'СМИ': 'СМИ',
        'РедСМИ': 'РедСМИ',
        'АПСТ': 'а пэ ээс тэ',
        'АППТ': 'а пэ пэ тэ',
        'ОПТОО': 'о пэ тэ о-о',
        'ОПСТ': 'о пэ эс тэ',
        'ОППТ': 'о пэ пэ тэ',
        'АСКФХ': 'а эс ка эф ха',
        'СОЮЗКФХ': 'СОЮЗ ка эф ха',
        'СОЮЗПОБ': 'СОЮЗ эп о бэ',
        'РСУ': 'эр су',
        'БСП': 'бэ ээс пэ',
        'ЦРБ': 'цэ эр бэ',
        'МУУЧ': 'МУУЧ',
        'МСЧ': 'эм ээс чэ',
        'ЦРБУХ': 'цэ эр бэ ух',
        'ЦБУХ': 'цэ бэ ух',
        'КЦ': 'ка цэ',
        'АТП': 'а тэ пэ',
        'ПАТП': 'пэ а тэ пэ',
        'ЦДН': 'цэ дэ эн',
        'НОТП': 'эн-о тэ пэ',
        'НОТК': 'эн-о тэ ка',
        'ОТД': 'о-тэ дэ',
        'КООП': 'КООП',

        'ЖСК': 'жэ эс ка',
        'ИТ': 'айти',
        'IT': 'айти',
        'ДНЛ': 'дэ эн эл',
        'НСК': 'эн эс ка',
        '3D': 'три дэ',
        'СТО': 'ээс тэ о',
        'ДНК': 'дэ эн ка',
        'СНГ': 'ээс эн гэ',
        'КС': 'ка ээс',
        'КЗ': 'ка зэ',
        'кз': 'кэ зэ',
        'УА': 'ю а',
        'уа': 'ю эй',

        'РФ': 'эр эф',
        'рф': 'эр эф',
        'РБ': 'эр бэ',
        'рб': 'эр бэ'
    }

    # Транскрипции отдельных букв
    transcription_letters = {
        # Согласные буквы
        'Б': 'бэ',
        'В': 'вэ',
        'Г': 'гэ',
        'Д': 'дэ',
        'Ж': 'жэ',
        'З': 'зэ',
        'Й': 'й',
        'К': 'ка',
        'Л': 'эл',
        'М': 'эм',
        'Н': 'эн',
        'П': 'пэ',
        'Р': 'эр',
        'С': 'ээс',
        'Т': 'тэ',
        'Ф': 'эф',
        'Х': 'ха',
        'Ц': 'цэ',
        'Ч': 'чэ',
        'Ш': 'ша',
        'Щ': 'ща',

        # Гласные буквы
        'А': 'а',
        'Е': 'е',
        'Ё': 'ё',
        'И': 'и',
        'О': 'о',
        'У': 'у',
        'Ы': 'ы',
        'Э': 'э',
        'Ю': 'ю',
        'Я': 'я',

        # На всякий случай
        'Ь': 'мягкий знак',
        'Ъ': 'твёрдый знак'
    }

    VOWELS = 'АЕЁИОУЫЭЮЯ'
    CONSONANTS = 'БВГДЖЗЙКЛМНПРСТФХЦЧШЩ'

    keywords_for_alphanumeric_addr = [
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


    def __init__(self):
        # Набор окончаний, необходимых для склонения чисел (для исключения их из обработки в конструкциях вида "число_одна_буква" и "число-одна_буква")
        self.exception_for_numerals_declination = set(NumeralsDeclination_ru.dict_of_ordinal_endings_and_pymorphy_tags.keys())

        self.framing_numbers_with_spaces = NumberWithSpacesCase()
        
        self.search_multiple_spaces = re.compile(r'\s{2,}')
        self.search_for_initials_in_names_re = re.compile(r'[А-ЯЁ](\.|\s|\.\s)[А-ЯЁ]\.?')

        # Поиск пробелов в названиях трасс с М, Р, А, источник: https://ru.wikipedia.org/wiki/Список_автомагистралей_России
        self.track_names_re = re.compile(r'(\s+[МРАMPA])\s+\-?(\d+)')

        # Для добавления дефиса между названием трассы и следующим числом, что бы они не сливались в одно число в случаях вида "Трасса М1 52км"
        self.track_names_with_next_number_re = re.compile(r'(\s+[МРАMPA])\s*\-?(\d+)(\s+\d+)')

        # Поиск пробелов во фразах вида "ключевое_слово число одна_буква"
        self.keyword_number_one_letter_re = re.compile(r'(({})[\.,!\?:;]*\s+\d+[\.,]?\-?\d*)\s+\-?([а-яёА-ЯЁa-zA-Z][\.,!\?:;\s]+)'.format(
                                                       '|'.join(self.keywords_for_alphanumeric_addr)))

        # Поиск пробелов во фразах вида "число одна_буква"
        self.number_one_letter_re = re.compile(r'(\s+\d+[\.,]?\-?\d*)\s+\-?([бгдеёжзлмнпртфхцчшщыюbcdfghjklmnpqrstvwxyz][\.,!\?:;\s]+)',
                                               flags=re.IGNORECASE)

        # Поиск пробелов во фразах вида "ключевое_слово одна_буква число"
        self.keyword_one_letter_number_re = re.compile(r'(({})[\.,!\?:;]*\s+[а-яёА-ЯЁa-zA-Z])\s+\-?(\d+[\.,]?\-?\d*)'.format(
                                                       '|'.join(self.keywords_for_alphanumeric_addr)))

        # Поиск пробелов во фразах вида "одна_буква число"
        self.one_letter_number_re = re.compile(r'(\s+[бгдеёжзлмнпртфхцчшщыюbcdfghjklmnpqrstvwxyz])\s+\-?(\d+[\.,]?\-?\d*)', flags=re.IGNORECASE)

        self.word_search_re = re.compile(r'(\W)')

        self.is_consonants = lambda text: all([symbol in self.CONSONANTS for symbol in text])
        self.is_vowels = lambda text: all([symbol in self.VOWELS for symbol in text])
        self.is_float_digit = lambda word: word.replace('.', '').replace(',', '').isdigit()
    

    def _tokenize(self, text, ignore_hyphen=False):
        ''' Разбиение строки на слова и отдельные символы, и объединение слов, у которых указано ударение символом '+' или которые записаны через '-'.

        1. text - строка с текстом для обработки
        2. ignore_hyphen - True: разделять слова, записанные через дефис
        3. возвращает список слов '''

        words = self.word_search_re.split(text)
        words = [word for word in words if word]

        is_extended_alpha = lambda word: word.replace('+', '').replace('-', '').isalpha()
        is_extended_alnum = lambda word: word.replace('+', '').replace('-', '').isalnum()
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
            

            elif words[i] in ['-', '.', ','] and is_extended_alnum(words[i-1]) and is_extended_alnum(words[i+1]):
                words[i-1] += words[i] + words[i+1]
                del words[i:i+2]

            else:
                i += 1
        
        # Если символ '+' в конце последнего слова (т.е. предыдущее слово является последовательностью букв)
        if words and words[-1] == '+':
            words[-2] += words[-1]
            del words[-1]
        return words


    def transcribe(self, text):
        ''' Поиск и транскрипция аббревиатур, инициалов в ФИО и буквенно-цифровых обозначений в русском тексте. Транскрипция выполняется в соответствии с
        транскрипцией/проищношением букв в русском алфавите.
        
        Аббревиатура - это последовательность из 2 и более заглавных букв, обрамлённая пробелами. Аббревиатуры транскрибируются по словарю и/или
        с помощью набора правил:
            - первая буква гласная, а остальные согласные
            - последняя буква гласная, а остальные согласные
            - первая и последняя буквы гласные, а остальные согласные
            - все буквы гласные или все буквы согласные

        Отдельно транскрибируются инициалы в ФИО вида 'И. О.', 'И.О.' и 'И О' и следующие буквенно-цифровые обозначения:
            - названия трасс, начинающиеся с 'М', 'Р', 'А'
            - фразы вида "Ключевое_слово Число Одна_буква"
            - фразы вида "Ключевое_слово Одна_буква Число"
            - фразы вида "Число Одна_буква" и "ЧислоОдна_буква"
            - фразы вида "Одна_буква Число" и "Одна_букваЧисло"

        Для правильной работы модуль должен работать после расшифровки сокращений и до перевода чисел в слова!

        1. text - строка с текстом для обработки
        2. возвращает обработанный текст '''

        if not text or text.isspace():
            return text

        remove_additional_spaces = [text[0].isspace(), text[-1].isspace()]

        # Обрамление всех чисел пробелами, с учётом возможного знака минус перед числом (т.е. дефиса), дефиса между числами и разделителей разрядов и
        # дробной части (т.е. с учётом точек и запятых)
        text = self.framing_numbers_with_spaces.frame_numbers(text)

        # Удаление нескольких подряд идущих пробелов и пробелов в начале и конце строки (использование text.strip() немного ломает перевод текста с фонемами
        # в последовательность чисел и обратно)
        text = self.search_multiple_spaces.sub(' ', text)

        # Добавление пробелов в конце строки для корректной обработки крайних буквенно-цифровых обозначений
        text += ' '

        # Удаление пробелов в названиях трасс с М, Р, А
        text = self.track_names_re.sub(r'\1\2', text)

        # Добавление дефиса между названием трассы и следующим числом, что бы они не сливались в одно число в случаях вида "Трасса М1 52км"
        text = self.track_names_with_next_number_re.sub(r'\1\2-\3', text)

        # Удаление пробелов во фразах вида "Ключевое_слово Число Одна_буква"
        text = self.keyword_number_one_letter_re.sub(r'\1\3', text)

        # Удаление пробелов во фразах вида "Число Одна_буква"
        text = self.number_one_letter_re.sub(r'\1\2', text)

        # Удаление пробелов во фразах вида "Ключевое_слово Одна_буква Число"
        text = self.keyword_one_letter_number_re.sub(r'\1\3', text)

        # Удаление пробелов во фразах вида "Одна_буква Число"
        text = self.one_letter_number_re.sub(r'\1\2', text)


        # Удаление пробелов в начале и конце строки, если их не было и они появились после обрамления всех чисел пробелами
        if remove_additional_spaces[0] != text[0].isspace():
            text = text[1:]
        if remove_additional_spaces[1] != text[-1].isspace():
            text = text[:-1]

        # Разбиение строки на слова и отдельные символы, и объединение слов, у которых указано ударение или которые записаны через -
        words = self._tokenize(text)

        # Исправление особенности работы регулярки для обрамления всех чисел пробелами: удаление пробела между экранирующим символом \ и
        # следующим за ним '-' или '(', если они есть
        i = 0
        while i < len(words) - 2:
            if words[i] == '\\' and words[i+1] == ' ' and words[i+2] in ['-', '(']:
                del words[i+1]
            else:
                i += 1


        # Основной цикл поиска и транскрипции аббревиатур, инициалов в ФИО и буквенно-цифровых обозначений
        i = 0
        while i < len(words):
            # Транскрипция инициалов в ФИО вида 'И. О.', 'И.О.' и 'И О'
            initials_in_names_match = self.search_for_initials_in_names_re.match(''.join(words[i:]))
            if initials_in_names_match:
                initials_in_name = initials_in_names_match.group(0)

                transcripted_initials = []
                for j, symbol in enumerate(initials_in_name):
                    if self.transcription_letters.get(symbol):
                        transcripted_initials.append(self.transcription_letters[symbol])
                        if j > 0 and initials_in_name[j-1] != ' ':
                            transcripted_initials[-1] = ' ' + transcripted_initials[-1]
                    else:
                        transcripted_initials.append(symbol)

                words[i:i+initials_in_names_match.regs[0][1]] = transcripted_initials
                i += initials_in_names_match.regs[0][1] - 1

            # Транскрипция аббревиатуры по словарю
            elif self.transcriptions.get(words[i]):
                words[i] = self.transcriptions[words[i]]

            # Транскрипция не словарных аббревиатур, но попадающих под правила:
            # - первая буква гласная, а остальные согласные
            # - последняя буква гласная, а остальные согласные
            # - первая и последняя буквы гласные, а остальные согласные
            # - все буквы гласные или все буквы согласные
            elif len(words[i]) > 1 and words[i].isalpha() and words[i].isupper():
                if (words[i][0] in self.VOWELS and self.is_consonants(words[i][1:])) \
                        or (words[i][-1] in self.VOWELS and self.is_consonants(words[i][:-1])) \
                        or (words[i][0] in self.VOWELS and words[i][-1] in self.VOWELS and self.is_consonants(words[i][1:-1])) \
                        or self.is_vowels(words[i]) or self.is_consonants(words[i]):

                    transcripted_word = []
                    for letter in words[i]:
                        if self.transcription_letters.get(letter):
                            transcripted_word.append(self.transcription_letters[letter])
                        else:
                            transcripted_word.append(letter)
                    words[i] = ' '.join(transcripted_word)

            # Конструкции вида "Одна_буква Число" и "Одна_букваЧисло"
            elif len(words[i]) > 1 and words[i][0].isalpha() and self.is_float_digit(words[i][1:]):
                if self.transcription_letters.get(words[i][0].upper()):
                    words[i] = self.transcription_letters[words[i][0].upper()] + ' ' + words[i][1:]
            
            # Конструкции вида "Число Одна_буква" и "ЧислоОдна_буква"
            elif len(words[i]) > 1 and self.is_float_digit(words[i][:-1]) and words[i][-1].isalpha() \
                    and words[i][-1] not in self.exception_for_numerals_declination:
                if self.transcription_letters.get(words[i][-1].upper()):
                    words[i] = words[i][:-1] + ' ' + self.transcription_letters[words[i][-1].upper()]

            # Конструкции вида "Одна_буква-Число"
            elif words[i].find('-') != -1 and len(words[i][:words[i].find('-')]) == 1 and words[i][:words[i].find('-')].isalpha() \
                    and self.is_float_digit(words[i][words[i].find('-')+1:]):
                letter = words[i][:words[i].find('-')].upper()
                if self.transcription_letters.get(letter):
                    words[i] = self.transcription_letters[letter] + ' ' + words[i][words[i].find('-')+1:]
            
            # Конструкции вида "Число-Одна_буква"
            elif words[i].find('-') != -1 and len(words[i][words[i].find('-')+1:]) == 1 and words[i][words[i].find('-')+1:].isalpha() \
                    and self.is_float_digit(words[i][:words[i].find('-')]) and words[i][words[i].find('-')+1:] not in self.exception_for_numerals_declination:
                letter = words[i][words[i].find('-')+1:].upper()
                if self.transcription_letters.get(letter):
                    words[i] = words[i][:words[i].find('-')] + ' ' + self.transcription_letters[letter]

            i += 1

        text = ''.join(words)
        return text




def main():
    test_texts = {
        '': '',
        '  ': '  ',
        'н31 и к2': 'эн 31 и ка 2',
        'улица 35-летия победы':
                'улица 35-летия победы',
        '1-й Вязовский проезд 19к4, 2901-е число, 31й, 29я, 37м':
                '1-й Вязовский проезд 19к4, 2901-е число, 31й, 29я, 37 эм',
        'ИП Клим В.О. и ЧТПЕ Ирцэн А. Ф.':
                'и пэ Клим вэ. о. чэ тэ пэ е Ирцэн а. эф.',
        'Агенство АБВГ вместе с АЗКЕ и ДЛШ пошли к ЕУА а затем к ПАБТУ':
                'Агенство а бэ вэ гэ вместе с а зэ ка е и дэ эл ша пошли к е у а а затем к ПАБТУ',
        'Компания: МСЧ ИТ в ФРГ':
                'Компания: эм ээс чэ айти в эф эр гэ',
        'ИП Иванов А.Д.':
                'и пэ Иванов а. дэ.',
        'Трассы М1 М-2, Р-153, Р89, дом89а, дом Б23 98.1Г 25п корпус г1, строение94т, л3.4, подвал32-в он шёл с 2сумками в дурдом-3, строение2.14в':
                'Трассы эм 1 эм 2, эр 153, эр 89, дом 89 а, дом бэ 23 98.1 гэ 25 пэ корпус гэ 1, строение 94 тэ, эл 3.4, подвал 32 вэ он шёл с ' + \
                '2 сумками в дурдом-3, строение 2.14 вэ',
        'Трасса М-53 94км, улица 20 летия 9б (ст Опора), дом 54б, стр39 литера9 павильон 2В':
                'Трасса эм 53- 94км, улица 20 летия 9 бэ (ст Опора), дом 54 бэ, стр 39 литера 9 павильон 2 вэ',
    }

    words_to_alphabetical_transcription_ru = WordsToAlphabeticalTranscription_ru()

    print('[i] Тестирование на {} примерах...'.format(len(test_texts)))
    elapsed_times = []
    error_result_tests = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        result = words_to_alphabetical_transcription_ru.transcribe(text)
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
