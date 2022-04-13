import re
import phonenumbers
from text_preparation.numbers_to_words_ru import one_number_to_words_ru
from text_preparation.flow.mutators.base import AbstractMutator
from text_preparation.flow.mutators.mixins import ReplaceSymbolsMixin

INTERNATIONAL_PHONE_FIRST_SYMBOL = '+'
INTERNATIONAL_PHONE_MIN_LENGTH = 11
DEFAULT_PHONE_REGION = "RU"


class PhoneVocalizer(AbstractMutator, ReplaceSymbolsMixin):
    """
    Класс для обработки номеров телефонов в текстовой строке
    Ищет номера по регулярному выражению, далее проверяет их корректность через  библиотеку phonenumbers
    Заменяет числа на их строковые значения без склонения, разделяя номер телефон на равные части

    Имеет возможность отключения форматирования, случае если перед номером телефона есть символ `!`
    Обрабатывает добавочные/внутренние номера

    @todo так как регулярными выражениями нельзя обработать все случаи, приходится использовать phonenumbers, который в
    свою очередь не умеет работать с номерами внутри страны (требуется всегда код города). Нужно доработать этот вопрос.
    """
    block_limit = 3
    disable_formatting_indicator = '!'
    phone_delimiter = ' - '

    dependencies = {
        'GeneralTextPreprocessing'
    }

    phone_regexp = re.compile(r"!?[\+]?[\d]{1,3}(?:[\s(]*[\d]{1,5}[)\s]*)?(?:[\s\-]?[0-9]{1,3})*"
                              r"(?:\s?(до(б|п)\.?|#|вн\.?)?\s?\d{2,5})?", flags=re.IGNORECASE)

    base_regexp = re.compile(r"[\+]?[\d]{1,3}(?:[\s(]*[\d]{1,5}[)\s]*)?(?:[\s\-]?[0-9]{1,3})*")
    additional_regexp = re.compile(r"(\s?(до(б|п)\.?|#|вн\.?)\s?\d{2,5})$", flags=re.IGNORECASE)
    number_regexp = re.compile(r"[0-9]+")

    replace_symbols_list = [
        (re.compile(r"доб\.?"), ' добавочный '),
        (re.compile(r"доп\.?"), ' дополнительный '),
        (re.compile(r"(вн\.?|#)"), ' внутренний '),
        (re.compile(r"[)(\.-]"), ' '),
    ]

    def __init__(self):
        self.disabled_auto_formatting = False

    def __call__(self, text: str) -> str:
        phones = self.phone_regexp.finditer(text)
        for match in phones:
            self.disabled_auto_formatting = (match.group()[0] == self.disable_formatting_indicator)
            base = self.base_regexp.search(match.group()).group()

            if not self.__is_valid_phone(base) and not self.disabled_auto_formatting:
                continue

            base = self.__processing_base_number(base)
            additional = self.additional_regexp.search(match.group())
            if additional is not None:
                additional = self.__processing_additional_number(additional.group().lower())

            phone = self.phone_delimiter.join([base, additional]) if additional is not None else base
            phone = phone + self.phone_delimiter
            text = text.replace(match.group(), self.__convert_numbers_to_text(phone), 1)

        return text

    def __processing_base_number(self, phone: str) -> str:
        """
        Обработка основной части номера
        """
        phone = self._replace_symbols(phone)
        if self.disabled_auto_formatting:
            return self.phone_delimiter.join(phone.split(' '))

        phone = re.sub(r"\s", '', phone)
        if phone[0] == INTERNATIONAL_PHONE_FIRST_SYMBOL:
            return self.phone_delimiter.join(["плюс", phone[1], phone[2:5], self.__number_slicing(phone[5:])])
        elif len(phone) >= INTERNATIONAL_PHONE_MIN_LENGTH:
            return self.phone_delimiter.join([phone[0], phone[1:4], self.__number_slicing(phone[4:])])
        else:
            return self.__number_slicing(phone)

    def __processing_additional_number(self, phone: str) -> str:
        """
        Обработка добавочного номера
        """
        phone = self._replace_symbols(phone)
        if self.disabled_auto_formatting:
            return phone

        number = re.search(r"\d", phone).group()
        return phone.replace(number, self.__number_slicing(number))

    def __number_slicing(self, phone: str):
        """
        Режем номер телефона на части, удобные для озвучивания
        """
        if len(phone) <= self.block_limit:
            return phone

        limit = len(phone) // 2
        return self.phone_delimiter.join([self.__number_slicing(phone[0:limit]), self.__number_slicing(phone[limit:])])

    def __convert_numbers_to_text(self, value: str) -> str:
        for match in self.number_regexp.findall(value):
            _, number = one_number_to_words_ru(match)
            value = value.replace(match, number, 1)

        return value

    @staticmethod
    def __is_valid_phone(phone: str) -> bool:
        """
        Фунция проверки валидности номера по базе данных номеров
        через библиотеку phonenumbers
        """
        try:
            return phonenumbers.is_valid_number(
                phonenumbers.parse(phone, DEFAULT_PHONE_REGION)
            )
        except Exception:
            return False