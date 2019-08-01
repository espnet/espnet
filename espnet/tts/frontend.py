# coding: utf-8

from random import random


class TTSFrontend(object):
    """Interface for TTS frontend

    Evert language-dependent TTS frontend should inherit the class and implement
    interface methods.
    """

    def num_vocab(self):
        """Returns number of vocabulary.

        Returns:
            int: number of vocabulary.
        """
        raise NotImplementedError()

    def text_to_sequence(self, text):
        """Convert text to its numeric representation.

        Args:
            text (str): Input text (e.g. character sequence)

        Returns:
            list: Sequence of token ID.
        """
        raise NotImplementedError()

    def sequence_to_text(self, sequence):
        """Convert numeric representation back to text

        Args:
            sequence (list): Input sequence of token ID.

        Returns:
            str: Text
        """
        raise NotImplementedError()


def get_frontend(name, *args, **kwargs):
    """Instantiate text processing frontend by short name (e.g. "en").

    Args:
        name (str): Name of frontend. "text" or "en".
        args (list): Arguments for frontend.
        kwargs (dict): Keyword arguments for frontend.

    Returns:
        TTSFrontend: Text processing frontend.
    """
    frontend_map = {
        "text": TEXT,
        "en": EN,
    }
    return frontend_map[name](*args, **kwargs)


class TEXT(TTSFrontend):
    """Generic less-language dependent text processing frontend.

    Args:
        cleaner_names (list): List of cleaner names. Available cleaners are:
            1. "english_cleaners" for English text
            2. "transliteration_cleaners" for non-English text that can be
                transliterated to ASCII using the Unidecode library
                (https://pypi.python.org/pypi/Unidecode)
            3. "basic_cleaners" if you do not want to transliterate (in this case,
                you should also update the symbols in symbols.py to match your data).
    """

    def __init__(self, cleaner_names=["english_cleaners"]):
        self.cleaner_names = cleaner_names

    def num_vocab(self):
        from espnet.tts.frontend_impl.text.symbols import symbols
        return len(symbols)

    def text_to_sequence(self, text):
        from espnet.tts.frontend_impl.text import text_to_sequence
        return text_to_sequence(text, self.cleaner_names)

    def sequence_to_text(self, sequence):
        from espnet.tts.frontend_impl.text import sequence_to_text
        return sequence_to_text(sequence)


def _maybe_get_arpabet(word, p, arpabet):
    try:
        phonemes = arpabet[word.lower()][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        return word

    return '{%s}' % phonemes if random() < p else word

# TODO: handle last word correctly


def _mix_pronunciation(text, p, arpabet):
    text = ' '.join(_maybe_get_arpabet(word, p, arpabet) for word in text.split(' '))
    return text


class EN(TEXT):
    """Text processing frontend for English.

    Args:
        mix_prob (float): char-to-phoneme mixing ratio. 0 (char only),
            1 (phoneme only), 0 ~ 1 (joint representation of char and phoneme).
            Phoneme is obtained by CMU pronunciation dictionary.

    Examples:
        Char representation

        >>> from espnet.tts.frontend import get_frontend
        >>> fe = get_frontend("en", mix_prob=0.0)
        >>> fe.text_to_sequence("hello world")
        [35, 32, 39, 39, 42, 64, 50, 42, 45, 39, 31, 60, 1]
        >>> fe.sequence_to_text(fe.text_to_sequence("hello world"))
        'hello world.~'

        Phoneme representation

        >>> fe = get_frontend("en", mix_prob=0.0)
        >>> fe.sequence_to_text(fe.text_to_sequence("hello world"))
        '{HH AH0 L OW1} .{W ER1 L D}~'

        On-the-fly char-phoneme mixing representation

        >>> fe = get_frontend("en", mix_prob=0.5)
        >>> fe.sequence_to_text(fe.text_to_sequence("hello world"))
        '{HH AH0 L OW1} .{W ER1 L D}~'
        >>> fe.sequence_to_text(fe.text_to_sequence("hello world"))
        'hello .{W ER1 L D}~'
        >>> fe.sequence_to_text(fe.text_to_sequence("hello world"))
        '{HH AH0 L OW1} world.~'

    .. note::
        Need to install `nltk` and dictionary `cmudict` to use mixed char-phoneme
        representation.

    """

    def __init__(self, mix_prob=0.0):
        super(EN, self).__init__(["english_cleaners"])
        self.mix_prob = mix_prob
        if mix_prob > 0:
            import nltk
            self.arpabet = nltk.corpus.cmudict.dict()
        else:
            self.arpabet = None

    def text_to_sequence(self, text):
        if self.mix_prob > 0:
            text = _mix_pronunciation(text, self.mix_prob, self.arpabet)
        return super(EN, self).text_to_sequence(text)
