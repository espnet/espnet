import re
from urllib.parse import urlparse

from polyglot.transliteration import Transliterator

from text_preparation.numbers_to_words_ru import one_number_to_words_ru
from text_preparation.flow.mutators.base import AbstractMutator
from text_preparation.flow.mutators.mixins import ReplaceSymbolsMixin, PauseSymbolsMixin


class URLVocalizer(AbstractMutator, ReplaceSymbolsMixin, PauseSymbolsMixin):
    ''' Класс для обработки ссылок и интернет адресов в текстовой строке. Преобразует ссылки в транслитерацию, оставляя только хост и часть пути.
    Заменяет числа на их строковые значения без склонения. '''

    cyrillic_domain_zones = '|'.join(['рф', 'ею', 'бг', 'рус', 'укр', 'бел', 'срб', 'орг', 'ком', 'сайт', 'дети', 'онлайн', 'москва', 'католик'])

    url_regexp_str = r"((?:https?:\/\/)?[\w\.\d-]{2,}\.(?:[a-z]{2,5}|" + cyrillic_domain_zones + \
                        r")((?:[\w\-\.,:;=\+\*_\/~\?!#@\$&|\[\]'\(\)]+)(?:[\w=\+\*_\/~\?!#@\$&|\[\]'\(\)]+))?)"
    url_regexp = re.compile(url_regexp_str, re.IGNORECASE)
    number_regexp = re.compile(r"[0-9]+")

    dependencies = {
        'GeneralTextPreprocessing'
    }

    replace_symbols_list = [
        (re.compile(r"^www\."), ' три дабл ю точка - '),
        ('.', ' точка '),
        (',', ' запятая '),
        (':', ' двоеточие - '),
        (';', ' точка с запятой - '),
        ('=', ' равно '),
        ('+', ' плюс '),
        ('*', ' звёздочка '),
        ('_', '-'),
        ('/', ' слэш - '),
        ('?', ' вопросительный знак '),
        ('!', ' восклицательный знак '),
        ('#', ' решётка '),
        ('@', ' собака '),
        ('$', ' доллар '),        
        ('&', ' и '),
        ('|', ' или ')
    ]

    transliterator = Transliterator(source_lang='en', target_lang='ru')
    alphabet_en = 'abcdefghijklmnopqrstuvwxyz'


    def __call__(self, text: str) -> str:
        urls = self.url_regexp.finditer(text)
        for match in urls:
            url = urlparse(match.group() if 'http' in match.group() else "http://%s" % match.group())

            result = self.__convert_numbers_to_text(url.hostname + url.path)
            result = self._replace_symbols(result)
            result = self.__transliterate(result)
            result = self.set_start_end_pause(result)

            text = text.replace(match.group(), result, 1)

        return text


    def __transliterate(self, value: str) -> str:
        return ''.join(map(lambda x: self.transliterator.transliterate(x) if x in self.alphabet_en else x, value))


    def __convert_numbers_to_text(self, value: str) -> str:
        for match in self.number_regexp.findall(value):
            _, number = one_number_to_words_ru(match)
            value = value.replace(match, " %s " % number, 1)

        return value