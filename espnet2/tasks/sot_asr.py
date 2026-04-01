"""SOT ASR Task for ESPnet2.

Extends ASRTask with:
- SOTWhisperModel (native OpenAI Whisper + uppercase min-CE)
- SOTWhisperPreprocessor (tiktoken-based, no HF dependency)

Uses the standard ESPnet build_model flow (frontend/specaug/normalize/
encoder/decoder/ctc) — the only changes are the model and preprocessor
class registrations.
"""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.sot_espnet_model import SOTWhisperModel
from espnet2.tasks.asr import (
    ASRTask,
    decoder_choices,
    encoder_choices,
    frontend_choices,
    normalize_choices,
    postencoder_choices,
    preencoder_choices,
    specaug_choices,
)
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import AbsPreprocessor, CommonPreprocessor
from espnet2.train.sot_preprocessor import SOTWhisperPreprocessor
from espnet2.utils.types import str2bool, str_or_none

# Add SOTWhisperModel to model choices
sot_model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetASRModel,
        sot_whisper=SOTWhisperModel,
    ),
    type_check=AbsESPnetModel,
    default="sot_whisper",
)

# Add SOTWhisperPreprocessor to preprocessor choices
sot_preprocessor_choices = ClassChoices(
    "preprocessor",
    classes=dict(
        default=CommonPreprocessor,
        sot_whisper=SOTWhisperPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="sot_whisper",
)


class SOTASRTask(ASRTask):
    """SOT ASR Task with native Whisper encoder/decoder."""

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        sot_model_choices,
        preencoder_choices,
        encoder_choices,
        postencoder_choices,
        decoder_choices,
        sot_preprocessor_choices,
    ]

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        super().add_task_arguments(parser)

        group = parser.add_argument_group(description="SOT Whisper related")
        group.add_argument(
            "--added_tokens_file",
            type=str,
            default=None,
            help="Path to text file with one added token per line",
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable]:
        """Build preprocessor.

        When preprocessor=sot_whisper, instantiate SOTWhisperPreprocessor
        directly with preprocessor_conf. Otherwise delegate to parent.
        """
        if not getattr(args, "use_preprocessor", True):
            return None

        preprocessor = getattr(args, "preprocessor", "default")

        if preprocessor == "sot_whisper":
            preprocessor_conf = getattr(args, "preprocessor_conf", {})
            return SOTWhisperPreprocessor(
                train=train,
                **preprocessor_conf,
            )
        else:
            return super().build_preprocess_fn(args, train)

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            return ("speech", "text")
        else:
            return ("speech",)

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()
