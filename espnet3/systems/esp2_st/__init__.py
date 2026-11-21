"""The ESPnet2 ST system package for ESPnet3.

Holds the system, its Inference, the BLEU metric and the shared helpers
under ``src``. Nothing is re-exported: ``inference`` has to import under a
bare install (``ci/check_inference_imports.py``), and reaching ``system``
from here would pull the training stack in with it.
"""
