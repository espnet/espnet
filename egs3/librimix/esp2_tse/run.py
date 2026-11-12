"""Run the LibriMix target speaker extraction recipe."""

from espnet3.systems.esp2_tse.system import ESP2TSESystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ESP2TSESystem)
