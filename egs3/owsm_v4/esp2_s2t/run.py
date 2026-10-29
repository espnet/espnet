"""Entry point for the OWSM v4 recipe."""

from espnet3.systems.esp2_s2t.system import ESP2S2TSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ESP2S2TSystem)
