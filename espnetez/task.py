# ESPnet-EZ Task class
# This class is a wrapper for Task classes to support custom datasets.
import argparse
from argparse import Namespace
import logging
from pathlib import Path
from typing import Any, Optional, Union, Dict
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.category_iter_factory import CategoryIterFactory
from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.iterators.multiple_iter_factory import MultipleIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.build_batch_sampler import build_batch_sampler
from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import AbsTask, IteratorOptions
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr_transducer import ASRTransducerTask
from espnet2.tasks.asvspoof import ASVSpoofTask
from espnet2.tasks.cls import CLSTask
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
from espnet2.tasks.gan_svs import GANSVSTask
from espnet2.tasks.gan_tts import GANTTSTask
from espnet2.tasks.gan_codec import GANCodecTask
from espnet2.tasks.hubert import HubertTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.mt import MTTask
from espnet2.tasks.s2st import S2STTask
from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.s2t_ctc import S2TTask as S2TCTCTask
from espnet2.tasks.slu import SLUTask
from espnet2.tasks.speechlm import SpeechLMTask
from espnet2.tasks.spk import SpeakerTask
from espnet2.tasks.ssl import SSLTask
from espnet2.tasks.st import STTask
from espnet2.tasks.svs import SVSTask
from espnet2.tasks.tts import TTSTask
from espnet2.tasks.tts2 import TTS2Task
from espnet2.tasks.uasr import UASRTask
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.abs_espnet_model import AbsESPnetModel

TASK_CLASSES = dict(
    asr=ASRTask,
    asr_transducer=ASRTransducerTask,
    asvspoof=ASVSpoofTask,
    cls=CLSTask,
    diar=DiarizationTask,
    enh_s2t=EnhS2TTask,
    enh_tse=TargetSpeakerExtractionTask,
    enh=EnhancementTask,
    gan_svs=GANSVSTask,
    gan_tts=GANTTSTask,
    gan_codec=GANCodecTask,
    hubert=HubertTask,
    lm=LMTask,
    mt=MTTask,
    s2st=S2STTask,
    s2t=S2TTask,
    s2t_ctc=S2TCTCTask,
    slu=SLUTask,
    speechlm=SpeechLMTask,
    spk=SpeakerTask,
    ssl=SSLTask,
    st=STTask,
    svs=SVSTask,
    tts=TTSTask,
    tts2=TTS2Task,
    uasr=UASRTask,
)


def get_ez_task(task_name: str, use_custom_dataset: bool = False) -> AbsTask:
    """
    Retrieve a customized task class for the ESPnet-EZ framework.

    This function returns a task class based on the specified task name.
    If the `use_custom_dataset` flag is set to True, a version of the task
    class that supports custom datasets will be returned. The returned class
    inherits from the appropriate base task class and may be extended with
    additional functionality.

    Args:
        task_name (str): The name of the task to retrieve. This must be one of
            the keys defined in the `TASK_CLASSES` dictionary, such as 'asr',
            'mt', 'tts', etc.
        use_custom_dataset (bool, optional): A flag indicating whether to use
            a version of the task class that supports custom datasets. Defaults
            to False.

    Returns:
        AbsTask: An instance of the task class corresponding to the provided
        `task_name`. If `use_custom_dataset` is True, the returned class will
        be capable of handling custom datasets.

    Raises:
        KeyError: If `task_name` is not found in the `TASK_CLASSES` dictionary.

    Examples:
        >>> asr_task = get_ez_task("asr")
        >>> custom_asr_task = get_ez_task("asr", use_custom_dataset=True)

        >>> mt_task = get_ez_task("mt")
        >>> custom_mt_task = get_ez_task("mt", use_custom_dataset=True)

    Note:
        The task classes are designed to be used within the ESPnet-EZ framework,
        which allows for flexibility in handling various speech and language tasks.
        Ensure that the required dependencies for the specific task are properly
        installed and configured.
    """
    task_class = TASK_CLASSES[task_name]

    if use_custom_dataset:
        return get_ez_task_with_dataset(task_name)

    class ESPnetEZTask(task_class):
        build_model_fn = None

        @classmethod
        def build_model(cls, args=None):
            return task_class.build_model(args=args)

    return ESPnetEZTask


@typechecked
def get_espnet_model(task_name: str, config: Union[Dict, DictConfig]) -> AbsESPnetModel:
    task = get_ez_task(task_name)
    default_config = task.get_default_config()
    default_config.update(model_config.model)
    espnet_model = task.build_model(Namespace(**default_config))
    return espnet_model
