"""Preserve the SpeechLM import path after moving its sources into ESPnet3."""

from espnet3.systems.speechlm import src as _src

__path__ = _src.__path__
