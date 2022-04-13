from text_preparation.flow.executor import FlowExecutor
from text_preparation.flow.mutators import GeneralTextPreprocessing

from text_preparation.flow.mutators.ru import URLVocalizer as URLVocalizerRu
from text_preparation.flow.mutators.ru import PhoneVocalizer as PhoneVocalizerRu

executor = FlowExecutor(
    GeneralTextPreprocessing,
    URLVocalizerRu,
    PhoneVocalizerRu
)

__all__ = [executor]
