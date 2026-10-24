"""Run AN4 LM statistics and training through the shared ESPnet3 launcher."""

from espnet3.systems.esp2_lm.system import ESP2LMSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ESP2LMSystem)
