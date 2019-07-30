# coding: utf-8


class TTSFrontend(object):
    """Interface for TTS frontend"""

    def num_vocab(self):
        raise NotImplementedError()

    def text_to_sequence(self, text):
        raise NotImplementedError()

    def sequence_to_text(self, sequence):
        raise NotImplementedError()


def get_frontend(name, *args, **kwargs):
    frontend_map = {
        "text": TEXT,
        "en": EN,
    }
    return frontend_map[name](*args, **kwargs)


class TEXT(TTSFrontend):
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


class EN(TEXT):
    def __init__(self):
        super(EN, self).__init__(["english_cleaners"])
