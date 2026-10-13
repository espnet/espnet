"""The OWSM (s2t) system package for ESPnet3.

Holds the system, its Inference, metrics and tokenizer for ESPnet2's ``s2t``
and ``s2t_ctc`` tasks. Nothing is re-exported: ``inference`` has to import
under a bare install (``ci/check_inference_imports.py``), and reaching
``system`` from here would pull the training stack in with it.
"""
