from argparse import Namespace

import yaml

from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asvspoof import ASVSpoofTask
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
from espnet2.tasks.enh import EnhancementTask
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


from argparse import Namespace

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


def get_task_class(task_name):
    """Returns the task class for the given task name.

    Args:
        task_name: The name of the task.

    Returns:
        The task class.
    """
    return TASK_CLASSES[task_name]


def load_yaml(path):
    with open(path, "r") as f:
        yml = yaml.load(f, Loader=yaml.Loader)
    return yml


def update_config(config, yml):
    """Updates the config with the given yaml.

    Args:
        config: The config to be updated.
        yml: The yaml to be updated.
    """
    config = vars(config)
    for k, v in yml.items():
        if isinstance(v, dict):
            if k not in config:
                config[k] = {}
            update_config(config[k], v)
        else:
            config[k] = v
    return Namespace(**config)
