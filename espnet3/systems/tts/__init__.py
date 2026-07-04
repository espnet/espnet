"""TTS system package for ESPnet3.

Exposes the generic TTS system, GAN-TTS trainer, and Lightning module
adapter shared across TTS recipes.
"""

from espnet3.systems.tts.system import TTSSystem

__all__ = ["TTSSystem"]
