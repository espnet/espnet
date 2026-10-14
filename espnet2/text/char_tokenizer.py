import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class CharTokenizer(AbsTokenizer):
    @typechecked
    def __init__(
        self,
        non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
        nonsplit_symbols: Optional[Iterable[str]] = None,
    ):
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        self.nonsplit_symbols = (
            set()
            if nonsplit_symbols is None
            else set([sym.split(":")[0] for sym in nonsplit_symbols])
        )
        # An empty symbol, e.g. from a blank line of the file, matches at every
        # position and text2tokens would never advance.
        self.non_linguistic_symbols.discard("")
        self.nonsplit_symbols.discard("")
        # A shorter symbol can be a prefix of a longer one. Set order is not
        # stable, so the longer symbol is tried first. The sets stay fixed
        # after init.
        self._ordered_symbols = tuple(
            sorted(
                self.non_linguistic_symbols.union(self.nonsplit_symbols),
                key=len,
                reverse=True,
            )
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f'nonsplit_symbols="{self.nonsplit_symbols}"'
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        removed = False
        tokens = []
        while len(line) != 0:
            for w in self._ordered_symbols:
                if line.startswith(w):
                    if (
                        w in self.nonsplit_symbols
                        or not self.remove_non_linguistic_symbols
                    ):
                        tokens.append(line[: len(w)])
                    else:
                        removed = True
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = self.space_symbol
                tokens.append(t)
                line = line[1:]
        if removed:
            # Removing a symbol between two spaces would leave two space tokens,
            # which a character error rate counts as an error. Word tokenization
            # ignores such spaces, so do the same here.
            tokens = self._merge_spaces(tokens)
        return tokens

    def _merge_spaces(self, tokens: List[str]) -> List[str]:
        merged = []
        for t in tokens:
            if t == self.space_symbol and (not merged or merged[-1] == t):
                continue
            merged.append(t)
        if merged and merged[-1] == self.space_symbol:
            merged.pop()
        return merged

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)
