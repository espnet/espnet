# -*- coding: utf-8 -*-
# pylint: disable=W0614

'''
Обрамление всех чисел и шаблонов с числами пробелами в русском и английском тексте.
'''

from typing import *
import re


StringRange = Tuple[int, int]

LETTERS = "a-zа-яё"
LETTERS_RE = f"[{LETTERS}]"
NON_LETTERS_RE = f"[^{LETTERS}\\d]"  # note '\d'
CURRENCIES_RE = "[$€₽£¥]"
DATE_NUMBER_SUFFIXES = "(?:ого|ому|его|летия)"

DASHED_WORD_RE = \
    f"""( {LETTERS_RE}                            # одна буква
        | {LETTERS_RE}([{LETTERS}-])*{LETTERS_RE} # две или несколько букв (с, возможно, `-` посредине)
        )
    """

POSITIVE_NUMBERS_RE = \
    f"""
        \\d+([.,:-]\\d+)*({CURRENCIES_RE})?
    """

TAIL_PUNCTUATION_RE = "[.;,:!]"


class CaseMatchingMixin:
    _case_group = set()

    def get_match_position(self, text: str) -> Optional[StringRange]:
        match = self._regexp_compiled.search(text)
        if match:
            for group in self._case_group:
                match_pos = match.span(group)
                if match_pos != (-1, -1):
                    return match_pos
        return None

    def frame_numbers(self, text: str) -> str:
        ''' Обрамление всех чисел и шаблонов с числами пробелами в русском и английском тексте. Поддерживаются следующие шаблоны (в [] указано
        перечисление, т.е. один из символов между |):
        1. [-|(]число[-|:|.|,|число][.|,|:|;|!|?|)|$|€|₽|£|¥]
        2. [1-2буквы|-]число[-|:|.|,|число][-|1-2буквы]
        3. [слово-]число[-|:|.|,|число][-слово]

        1. text - строка с текстом для обработки
        2. возвращает обработанный текст '''

        # match_pos = get_match_position(text)
        match_pos = self.get_match_position(text)
        if match_pos:
            start, end = match_pos
            result = text[:start]
            if start > 0 and text[start - 1] != ' ':  # TODO any space symbol!
                result += ' '
            result += text[start:end]
            if end < len(text) and text[end] != ' ':
                result += ' '
            result += self.frame_numbers(text[end:])
            return result
        else:
            return text


class BaseCase:
    def __init__(self):
        self._regexp_compiled = re.compile(self._regexp_raw, re.VERBOSE | re.IGNORECASE)

    def __str__(self):
        return self._regexp_raw


class SimpleNumberCase(BaseCase):
    _regexp_raw = \
        f"""(?P<SimpleNumberCase>
                    ( \\(? -?{POSITIVE_NUMBERS_RE} \\)? )
                    {TAIL_PUNCTUATION_RE}?
        )"""


# Прилипание ноля, одной, двух букв, а также спец.суффикса
#
# Распадается на три случая: буквы вначале, буквы в конце, буквы вокруг.
#
# При этом, если есть много букв до/после, их надо отбросить, например,
#
#   `строение29т` -> `строение 29т`
#   `А12шоссе` -> `А12 шоссе`
#
# Для учёта таких случаев используются три регулярки,
# которые проверяют окружение числа.

# Подслучай "1-2 буквы вокруг числа"
class NumberWithSuffixLetterSubcase1Case(BaseCase):
    _regexp_raw = \
        f"""(?:^|{NON_LETTERS_RE})
            (?P<NumberWithSuffixLetterSubcase1Case>
                    {LETTERS_RE}{LETTERS_RE}?{POSITIVE_NUMBERS_RE}(({LETTERS_RE}{LETTERS_RE}?)|{DATE_NUMBER_SUFFIXES})
                    {TAIL_PUNCTUATION_RE}?
            )
            (?:$|{NON_LETTERS_RE})
        """


# Подслучай "1-2 буквы после числа"; тогда слева допустимы любые символы, кроме '-'
class NumberWithSuffixLetterSubcase2Case(BaseCase):
    _regexp_raw = \
        f"""(?P<NumberWithSuffixLetterSubcase2Case>
                    -?{POSITIVE_NUMBERS_RE}(({LETTERS_RE}{LETTERS_RE}?)|{DATE_NUMBER_SUFFIXES})
                    {TAIL_PUNCTUATION_RE}?
            )
            (?:$|{NON_LETTERS_RE})
        """


class NumberWithSuffixLetterSubcase3Case(BaseCase):
    _regexp_raw = \
        f""" (^|{NON_LETTERS_RE})
            (?P<NumberWithSuffixLetterSubcase3Case>
                    {LETTERS_RE}{LETTERS_RE}?{POSITIVE_NUMBERS_RE}
                    {TAIL_PUNCTUATION_RE}?
            )
        """


class NumberWithDashCase(BaseCase):
    _regexp_raw = \
        f"""(?:^|{LETTERS_RE}??) # для случая 'дома90-ых' разрешим non-greedy (??) буквы перед числом
            (?P<NumberWithDashCase>
                    ( ( {DASHED_WORD_RE}-{POSITIVE_NUMBERS_RE}-{DASHED_WORD_RE} )  # буквы перед и после числа
                    | ( {DASHED_WORD_RE}-{POSITIVE_NUMBERS_RE}                  )  # буквы перед числом
                    | (                  {POSITIVE_NUMBERS_RE}-{DASHED_WORD_RE} )  # буквы после числа
                    )
                    {TAIL_PUNCTUATION_RE}?
            )
            (?:$|{NON_LETTERS_RE})
        """


class NumberWithSpacesCase(BaseCase, CaseMatchingMixin):
    def __init__(self):
        # note: order is significant!
        cases = [NumberWithDashCase, NumberWithSuffixLetterSubcase1Case, NumberWithSuffixLetterSubcase2Case, NumberWithSuffixLetterSubcase3Case, SimpleNumberCase]
        self._case_group = set()
        regex_parts = []
        for case_class in cases:
            case_obj = case_class()
            regex_parts.append(f"({case_obj})")
            self._case_group.add(case_class.__name__)
        self._regexp_raw = '|'.join(regex_parts)
        super().__init__()


def interactive():
    # Тестирование на вводе пользователя
    while True:
        text = input('\n[i] Введите текст: ')
        start_time = time.time()
        result = NumberWithSpacesCase().frame_numbers(text)
        elapsed_time = time.time() - start_time
        print("[i] Результат: '{}'".format(result))
        print('[i] Время обработки {:.6f} с или {:.2f} мс'.format(elapsed_time, elapsed_time * 1000))


if __name__ == '__main__':
    interactive()
