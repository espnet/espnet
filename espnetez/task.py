# ESPnet-Easy Task class
# This class is a wrapper for Task classes to support custom datasets and models.
import numpy as np
import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader
from typeguard import check_argument_types
import torch

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import AbsTask, IteratorOptions
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asvspoof import ASVSpoofTask
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
from espnet2.tasks.gan_svs import GANSVSTask
from espnet2.tasks.gan_tts import GANTTSTask
from espnet2.tasks.hubert import HubertTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.mt import MTTask
from espnet2.tasks.slu import SLUTask
from espnet2.tasks.st import STTask
from espnet2.tasks.svs import SVSTask
from espnet2.tasks.tts import TTSTask
from espnet2.tasks.uasr import UASRTask
from espnet2.samplers.build_batch_sampler import build_batch_sampler

TASK_CLASSES = dict(
    asr=ASRTask,
    asvspoof=ASVSpoofTask,
    diar=DiarizationTask,
    enh_s2t=EnhS2TTask,
    enh_tse=TargetSpeakerExtractionTask,
    enh=EnhancementTask,
    gan_svs=GANSVSTask,
    gan_tts=GANTTSTask,
    hubert=HubertTask,
    lm=LMTask,
    mt=MTTask,
    slu=SLUTask,
    st=STTask,
    svs=SVSTask,
    tts=TTSTask,
    uasr=UASRTask,
)


def get_easy_task(task_name: str) -> AbsTask:
    task_class = TASK_CLASSES[task_name]

    class ESPnetEasyTask(task_class):
        model = None

        @classmethod
        def build_model(cls, args=None):
            if cls.model is not None:
                return cls.model
            else:
                return task_class.build_model(args=args)

    return ESPnetEasyTask
