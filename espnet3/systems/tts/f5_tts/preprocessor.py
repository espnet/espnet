"""F5 pinyin preprocessor for zh+en training (repo-exact tokenization).

Why a dedicated preprocessor instead of ``CommonPreprocessor`` + a ``f5_pinyin``
g2p: F5's fixed ``Emilia_ZH_EN_pinyin/vocab.txt`` has **no ``<unk>``** and maps any
unknown token to index 0, whereas ESPnet's ``TokenIDConverter`` *requires* a
``<unk>`` symbol. To reproduce the public repo exactly (and stay compatible with
F5's pretrained checkpoints) we use F5's own ``{token: id}`` mapping with the
unknown-> 0 fallback.

It only tokenizes ``text`` (-> int64 ids) and passes the waveform through; the mel
is produced in the model by ``VocoderMelSpec``, same as the char recipe.
A plain callable (not an ``AbsPreprocessor``), so ESPnet3 calls
``preprocessor(sample)``.
"""

from __future__ import annotations

from espnet3.systems.tts.f5_tts.pinyin import load_vocab_char_map, text_to_pinyin_ids


class F5PinyinPreprocessor:
    """Tokenize text with F5's zh+en pinyin scheme using F5's fixed vocab."""

    def __init__(
        self,
        vocab_file: str,
        text_name: str = "text",
        train: bool = True,
    ):
        """Configure the preprocessor.

        Args:
            vocab_file: F5's ``vocab.txt`` (e.g.
                ``Emilia_ZH_EN_pinyin/vocab.txt``); line number = token id,
                space at index 0.
            text_name: Sample dict key holding the raw transcript.
            train: Accepted for the collect_stats train/valid toggle
                (no-op here).

        Example:
            .. code-block:: yaml

                preprocessor:
                  _target_: espnet3.systems.tts.f5_tts.preprocessor.F5PinyinPreprocessor
                  vocab_file: /path/to/Emilia_ZH_EN_pinyin/vocab.txt

        Note:
            The vocabulary is read once at construction, so ``vocab_file`` must
            be the same one the checkpoint was trained with: it fixes the token
            ids, and a different file silently remaps them.
        """
        self.vocab_char_map = load_vocab_char_map(vocab_file)
        self.text_name = text_name
        self.train = train

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in F5's fixed vocabulary.

        Returns:
            Line count of ``vocab_file``.

        Note:
            This is the value to use as the model's ``idim`` when training
            against F5's fixed vocabulary instead of an ESPnet token list.
        """
        return len(self.vocab_char_map)

    def __call__(self, data: dict) -> dict:
        """Replace the raw transcript in ``data`` with pinyin token ids.

        Args:
            data: Sample dict holding the raw transcript under ``text_name``.

        Returns:
            The same dict, with that entry replaced by an ``int64`` id array.
            Every other key is left untouched.

        Example:
            .. code-block:: python

                >>> prep({"text": "abc"})["text"]
                array([1, 2, 3])

        Note:
            Mutates and returns the dict it is given rather than copying.
            Unknown tokens become 0, F5's literal space, so out-of-vocab text
            degrades quietly instead of raising.
        """
        data[self.text_name] = text_to_pinyin_ids(
            data[self.text_name], self.vocab_char_map
        )
        return data
