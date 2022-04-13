#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
#     DATE : 17.06.2020
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Предназначен для перевода чисел в слова. Поддерживаются положительные и отрицательные целые и дробные числа,
порядковые числа и денежные единицы (фунты £ и доллары $). Основано на https://github.com/keithito/tacotron
'''

import re
import inflect


inflect_engine = inflect.engine()
comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
number_re = re.compile(r'[0-9]+')


def __remove_commas(m):
    return m.group(1).replace(',', '')


def __expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def __expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # неподдерживаемый формат
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def __expand_ordinal(m):
    return inflect_engine.number_to_words(m.group(0))


def __expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + inflect_engine.number_to_words(num % 100)
        elif num % 100 == 0:
            return inflect_engine.number_to_words(num // 100) + ' hundred'
        else:
            return inflect_engine.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return inflect_engine.number_to_words(num, andword='')


def numbers_to_words_en(text):
    ''' Поддерживаются положительные и отрицательные целые и дробные числа, порядковые и денежные единицы (фунты £ и доллары $). '''
    text = re.sub(comma_number_re, __remove_commas, text)
    text = re.sub(pounds_re, r'\1 pounds', text)
    text = re.sub(dollars_re, __expand_dollars, text)
    text = re.sub(decimal_number_re, __expand_decimal_point, text)
    text = re.sub(ordinal_re, __expand_ordinal, text)
    text = re.sub(number_re, __expand_number, text)
    return text
