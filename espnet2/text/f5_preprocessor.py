"""F5 pinyin preprocessor for zh+en training (repo-exact tokenization).

Why a dedicated preprocessor instead of ``CommonPreprocessor`` + a ``f5_pinyin``
g2p: F5's fixed ``Emilia_ZH_EN_pinyin/vocab.txt`` has **no ``<unk>``** and maps any
unknown token to index 0, whereas ESPnet's ``TokenIDConverter`` *requires* a
``<unk>`` symbol. To reproduce the public repo exactly (and stay compatible with
F5's pretrained checkpoints) we use F5's own ``{token: id}`` mapping with the
unknown-> 0 fallback.

It only tokenizes ``text`` (-> int64 ids) and passes the waveform through; the mel
is produced in the model by ``feats_extract: vocoder_mel``, same as the char recipe.
A plain callable (not an ``AbsPreprocessor``), so ESPnet3 calls
``preprocessor(sample)``.
"""

from __future__ import annotations

from espnet2.text.f5_pinyin import load_vocab_char_map, text_to_pinyin_ids


class F5PinyinPreprocessor:
    """Tokenize text with F5's zh+en pinyin scheme using F5's fixed vocab."""

    def __init__(
        self,
        vocab_file: str,
        text_name: str = "text",
        train: bool = True,
    ):
        """Args:
        vocab_file: F5's ``vocab.txt`` (e.g. ``Emilia_ZH_EN_pinyin/vocab.txt``);
            line number = token id, space at index 0.
        text_name: Sample dict key holding the raw transcript.
        train: Accepted for the collect_stats train/valid toggle (no-op here).
        """
        self.vocab_char_map = load_vocab_char_map(vocab_file)
        self.text_name = text_name
        self.train = train

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_char_map)

    def __call__(self, data: dict) -> dict:
        data[self.text_name] = text_to_pinyin_ids(
            data[self.text_name], self.vocab_char_map
        )
        return data
