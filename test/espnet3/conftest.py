# tests/conftest.py
import builtins
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def fake_espnet_modules(monkeypatch):
    espnet2 = types.ModuleType("espnet2")
    tasks_pkg = types.ModuleType("espnet2.tasks")
    train_pkg = types.ModuleType("espnet2.train")

    class AbsESPnetModel:
        pass

    train_abs_mod = types.ModuleType("espnet2.train.abs_espnet_model")
    train_abs_mod.AbsESPnetModel = AbsESPnetModel

    def make_task_class(name):
        class _Task:
            called_with = None

            @classmethod
            def get_default_config(cls):
                return {
                    "default_key": "keepme",
                    "a_tuple": (1, 2),
                    "some_conf": None,
                }

            @classmethod
            def build_model(cls, ns):
                cls.called_with = ns
                return f"{name}-MODEL"

        _Task.__name__ = f"{name}Task"
        return _Task

    name_map = {
        "asr": ("espnet2.tasks.asr", "ASRTask"),
        "asr_transducer": ("espnet2.tasks.asr_transducer", "ASRTransducerTask"),
        "asvspoof": ("espnet2.tasks.asvspoof", "ASVSpoofTask"),
        "diar": ("espnet2.tasks.diar", "DiarizationTask"),
        "enh": ("espnet2.tasks.enh", "EnhancementTask"),
        "enh_s2t": ("espnet2.tasks.enh_s2t", "EnhS2TTask"),
        "enh_tse": ("espnet2.tasks.enh_tse", "TargetSpeakerExtractionTask"),
        "gan_svs": ("espnet2.tasks.gan_svs", "GANSVSTask"),
        "gan_tts": ("espnet2.tasks.gan_tts", "GANTTSTask"),
        "hubert": ("espnet2.tasks.hubert", "HubertTask"),
        "lm": ("espnet2.tasks.lm", "LMTask"),
        "mt": ("espnet2.tasks.mt", "MTTask"),
        "s2st": ("espnet2.tasks.s2st", "S2STTask"),
        "s2t": ("espnet2.tasks.s2t", "S2TTask"),
        "slu": ("espnet2.tasks.slu", "SLUTask"),
        "spk": ("espnet2.tasks.spk", "SpeakerTask"),
        "st": ("espnet2.tasks.st", "STTask"),
        "svs": ("espnet2.tasks.svs", "SVSTask"),
        "tts": ("espnet2.tasks.tts", "TTSTask"),
        "uasr": ("espnet2.tasks.uasr", "UASRTask"),
    }

    sys.modules["espnet2"] = espnet2
    sys.modules["espnet2.tasks"] = tasks_pkg
    sys.modules["espnet2.train"] = train_pkg
    sys.modules["espnet2.train.abs_espnet_model"] = train_abs_mod

    for key, (modname, clsname) in name_map.items():
        m = types.ModuleType(modname)
        TaskCls = make_task_class(clsname.replace("Task", ""))
        setattr(m, clsname, TaskCls)
        sys.modules[modname] = m

    yield

    for k in list(sys.modules.keys()):
        if k.startswith("espnet2"):
            sys.modules.pop(k, None)
