from .audio_loader import ArkiveAudioReader, LhotseAudioReader
from .dialogue_loader import DialogueReader
from .text_loader import ArkiveTextReader, TextReader

ALL_DATA_LOADERS = {
    "lhotse_audio": LhotseAudioReader,
    "arkive_audio": ArkiveAudioReader,
    "text": TextReader,
    "arkive_text": ArkiveTextReader,
    "dialogue": DialogueReader,
}

__all__ = [
    "ALL_DATA_LOADERS",
]
