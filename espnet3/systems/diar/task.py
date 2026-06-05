"""Diarization task for ESPnet3.

Thin re-export so recipe configs can set
``task: espnet3.systems.diar.task.SortformerDiarizationTask`` mirroring the ASR
convention.
"""

from espnet2.tasks.diar_sortformer import SortformerDiarizationTask

__all__ = ["SortformerDiarizationTask"]
