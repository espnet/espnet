"""ESPnet2 GAN-TTS system package for ESPnet3.

Holds the system (``system.py``), the compatibility copy of the espnet2
GAN-TTS task (``task.py``) and the implementation modules (``src/``): the GAN
Lightning adapters, the x-vector extraction Runner/Provider pair and the
inference runner. The ``measure`` stage uses the shared
``espnet3.components.metrics.versa.VersaMetric``.
"""
