# ESPnet-Easy Task class
# This class is a wrapper for Task classes to support custom datasets and models.

from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr_transducer import ASRTransducerTask
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
from espnet2.tasks.s2st import S2STTask
from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.slu import SLUTask
from espnet2.tasks.spk import SpeakerTask
from espnet2.tasks.st import STTask
from espnet2.tasks.svs import SVSTask
from espnet2.tasks.tts import TTSTask
from espnet2.tasks.uasr import UASRTask

TASK_CLASSES = dict(
    asr=ASRTask,
    asr_transducer=ASRTransducerTask,
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
    s2st=S2STTask,
    s2t=S2TTask,
    slu=SLUTask,
    spk=SpeakerTask,
    st=STTask,
    svs=SVSTask,
    tts=TTSTask,
    uasr=UASRTask,
)


def get_easy_task(task_name: str) -> AbsTask:
    task_class = TASK_CLASSES[task_name]

    class ESPnetEasyTask(task_class):
        build_model_fn = None

        @classmethod
        def build_model(cls, args=None):
            if cls.build_model_fn is not None:
                return cls.build_model_fn(args=args)
            else:
                return task_class.build_model(args=args)

    return ESPnetEasyTask
