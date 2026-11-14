"""Run VOiCES ASR stages through the shared ESPnet3 launcher."""

from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ASRSystem)
