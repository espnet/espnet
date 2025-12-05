from .audio_loader import LhotseAudioReader, ArkiveAudioReader
from .text_loader import TextReader, ArkiveTextReader
from .dialogue_loader import DialogueReader

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
