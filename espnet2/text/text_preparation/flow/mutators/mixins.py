import re
from typing import Tuple, List, Union


class ReplaceSymbolsMixin:
    replace_symbols_list = [
        (re.compile(r"\s{2,}"), ' ')
    ]

    def _replace_symbols(self, text: str) -> str:
        symbols = self.get_replacing_symbols_list()
        for pattern, val in symbols:
            if isinstance(pattern, re.Pattern):
                text = pattern.sub(val, text)

            if isinstance(pattern, str):
                text = text.replace(pattern, val)

        return text

    def get_replacing_symbols_list(self) -> List[Tuple[Union[re.Pattern, str], str]]:
        return self.replace_symbols_list


class PauseSymbolsMixin:
    pause_symbol = ' - '

    def set_start_end_pause(self, value: str) -> str:
        if re.match(r"^ - ", value) is None:
            value = self.pause_symbol + value

        if re.match(r" - $", value) is None:
            value = value + self.pause_symbol

        return value
