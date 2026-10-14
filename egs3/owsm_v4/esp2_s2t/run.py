"""Entry point for the OWSM v4 recipe."""

from espnet3.systems.esp2_s2t.system import OWSMSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(OWSMSystem)
