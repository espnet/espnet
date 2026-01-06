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
from espnet2.tasks.spk import SpeakerTask
from espnet2.tasks.st import STTask
from espnet2.tasks.svs import SVSTask
from espnet2.tasks.tts import TTSTask
from espnet2.tasks.uasr import UASRTask

__all__ = [
    "ASRTask",
    "ASRTransducerTask",
    "ASVSpoofTask",
    "DiarizationTask",
    "EnhancementTask",
    "EnhS2TTask",
    "TargetSpeakerExtractionTask",
    "GANSVSTask",
    "GANTTSTask",
    "HubertTask",
    "LMTask",
    "MTTask",
    "S2STTask",
    "S2TTask",
    "SpeakerTask",
    "STTask",
    "SVSTask",
    "TTSTask",
    "UASRTask",
]
