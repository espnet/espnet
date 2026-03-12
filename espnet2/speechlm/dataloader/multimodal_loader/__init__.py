from .audio_loader import ArkiveAudioReader, KaldiAudioReader, LhotseAudioReader
from .dialogue_loader import ArkiveDialogueLoader, DialogueReader
from .text_loader import ArkiveTextReader, TextReader

ALL_DATA_LOADERS = {
    "lhotse_audio": LhotseAudioReader,
    "arkive_audio": ArkiveAudioReader,
    "text": TextReader,
    "arkive_text": ArkiveTextReader,
    "dialogue": DialogueReader,
    "arkive_dialogue": ArkiveDialogueLoader,
    "kaldi_audio": KaldiAudioReader,
}

__all__ = [
    "ALL_DATA_LOADERS",
]
