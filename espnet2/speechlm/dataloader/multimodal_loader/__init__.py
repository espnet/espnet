from .audio_loader import KaldiAudioReader, LhotseAudioReader, OmniIOAudioReader
from .dialogue_loader import DialogueReader, OmniIODialogueLoader
from .text_loader import OmniIOTextReader, TextReader

ALL_DATA_LOADERS = {
    "lhotse_audio": LhotseAudioReader,
    "omniio_audio": OmniIOAudioReader,
    "text": TextReader,
    "omniio_text": OmniIOTextReader,
    "dialogue": DialogueReader,
    "omniio_dialogue": OmniIODialogueLoader,
    "kaldi_audio": KaldiAudioReader,
}

__all__ = [
    "ALL_DATA_LOADERS",
]
