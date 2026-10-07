"""The ASR system's Inference, with a stand-in for Speech2Text."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.api.inference import Audio
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.esp2_asr.inference import Inference


class FakeSpeech2Text:
    """Returns the n-best list Speech2Text does, remembering its input."""

    def __init__(self, fs="16k", per_speaker=False, **kwargs):
        self.asr_train_args = SimpleNamespace(frontend_conf={"fs": fs})
        self.per_speaker = per_speaker
        self.kwargs = kwargs
        self.seen = None

    def __call__(self, speech):
        self.seen = speech
        if isinstance(speech, list):
            return self.batch_decode(speech)
        best = ("hello world", ["hello", "world"], [1, 2], None)
        nbest = [best, ("hello", ["hello"], [1], None)]
        return [nbest, nbest] if self.per_speaker else nbest

    def batch_decode(self, speeches):
        self.batches = getattr(self, "batches", []) + [len(speeches)]
        return [[(f"utt{i}", None, None, None)] for i in range(len(speeches))]


def test_transcribes_the_best_hypothesis():
    backend = FakeSpeech2Text()
    model = Inference(backend)
    out = model(np.zeros(16000, dtype=np.float32))
    assert out == {"text": "hello world"}
    assert isinstance(backend.seen, np.ndarray) and backend.seen.dtype == np.float32


def test_first_speaker_of_a_joint_model():
    assert Inference(FakeSpeech2Text(per_speaker=True))(np.zeros(16))["text"] == (
        "hello world"
    )


def test_sample_rate_comes_from_the_frontend_config():
    assert Inference(FakeSpeech2Text(fs="16k")).sample_rate == 16000
    assert Inference(FakeSpeech2Text(fs=8000)).sample_rate == 8000
    # an empty frontend_conf, as mini_an4 trains with: DefaultFrontend's 16 kHz
    silent = FakeSpeech2Text()
    silent.asr_train_args = SimpleNamespace(frontend_conf={})
    assert Inference(silent).sample_rate == 16000
    with pytest.raises(TypeError, match="cannot tell the rate"):
        Inference(SimpleNamespace()).sample_rate  # not a Speech2Text at all


def test_audio_is_resampled_to_the_frontend_rate():
    backend = FakeSpeech2Text(fs=8000)
    Inference(backend)(Audio(np.zeros(16000, dtype=np.float32), 16000))
    assert len(backend.seen) == 8000


def test_from_pretrained_takes_a_directory_or_a_tag(tmp_path, monkeypatch):
    import espnet3.systems.base.backend_inference as base

    backend = FakeSpeech2Text()
    calls = []

    def locate_pack(tag_or_dir):
        calls.append(("locate", str(tag_or_dir)))
        return tmp_path

    def load_model(pack_dir, *, device=None, overrides=None):
        calls.append(("build", str(pack_dir), device, overrides))
        return backend

    monkeypatch.setattr(base, "locate_pack", locate_pack)
    monkeypatch.setattr(base, "load_model", load_model)

    assert Inference.from_pretrained(tmp_path, device="cpu").backend is backend
    assert Inference.from_pretrained("org/model", device="cuda:0").backend is backend
    # constructor arguments replace the packed ones, as in ESPnet2
    assert Inference.from_pretrained(tmp_path, beam_size=1).backend is backend
    assert calls == [
        ("locate", str(tmp_path)),
        ("build", str(tmp_path), "cpu", {}),
        ("locate", "org/model"),
        ("build", str(tmp_path), "cuda:0", {}),
        ("locate", str(tmp_path)),
        ("build", str(tmp_path), "cpu", {"beam_size": 1}),
    ]


def test_builds_a_speech2text_from_its_own_arguments(monkeypatch):
    import espnet2.bin.asr_inference as asr_inference

    monkeypatch.setattr(asr_inference, "Speech2Text", FakeSpeech2Text)
    model = Inference(asr_train_config="c.yaml", asr_model_file="m.pth", beam_size=3)
    assert model.backend.kwargs == {
        "device": "cpu",
        "asr_train_config": "c.yaml",
        "asr_model_file": "m.pth",
        "beam_size": 3,
    }
    assert model(np.zeros(16))["text"] == "hello world"
    with pytest.raises(TypeError, match="but a built one was too"):
        Inference(FakeSpeech2Text(), beam_size=3)


def test_builds_the_transducer_speech2text_when_told(monkeypatch):
    import espnet2.bin.asr_transducer_inference as transducer

    monkeypatch.setattr(transducer, "Speech2Text", FakeSpeech2Text)
    model = Inference(
        backend_class="espnet2.bin.asr_transducer_inference.Speech2Text",
        asr_train_config="c.yaml",
        return_decoded_hyp=True,
    )
    assert model.backend.kwargs == {
        "device": "cpu",
        "asr_train_config": "c.yaml",
        "return_decoded_hyp": True,
    }
    assert model(np.zeros(16))["text"] == "hello world"


def test_the_infer_stage_runner_calls_it_as_it_is():
    """InferenceRunner needs no output_fn: the contract's mapping is written."""
    model = Inference(FakeSpeech2Text())
    dataset = {
        i: {"speech": np.zeros(160, dtype=np.float32), "text": "ref"} for i in range(3)
    }
    assert InferenceRunner.forward(0, dataset=dataset, model=model) == {
        "utt_id": "0",
        "text": "hello world",
    }
    batched = InferenceRunner.forward([1, 2], dataset=dataset, model=model)
    assert batched == [  # one batch_decode, ids from the index
        {"utt_id": "1", "text": "utt0"},
        {"utt_id": "2", "text": "utt1"},
    ]


def test_the_infer_stage_provider_builds_it_from_inference_yaml(monkeypatch):
    import espnet2.bin.asr_inference as asr_inference

    monkeypatch.setattr(asr_inference, "Speech2Text", FakeSpeech2Text)
    config = OmegaConf.create(
        {
            "device": "cpu",
            "model": {
                "_target_": "espnet3.systems.esp2_asr.inference.Inference",
                "asr_train_config": "exp/config.yaml",
                "asr_model_file": "exp/model.pth",
            },
        }
    )
    model = InferenceProvider.build_model(config)
    assert isinstance(model, Inference)
    assert model.backend.kwargs["asr_train_config"] == "exp/config.yaml"


def test_a_batch_is_one_beam_search_when_the_backend_can():
    backend = FakeSpeech2Text()
    model = Inference(backend)
    items = [{"speech": np.zeros(16, dtype=np.float32)}] * 3
    assert [o["text"] for o in model.batch(items)] == ["utt0", "utt1", "utt2"]
    assert backend.batches == [3]
    assert model.batch(items[:1]) == [{"text": "hello world"}]  # one item: run

    class NoBatch(FakeSpeech2Text):
        batch_decode = None  # the transducer Speech2Text has none

    assert [o["text"] for o in Inference(NoBatch()).batch(items)] == ["hello world"] * 3


def test_from_pretrained_returns_the_bundles_own_inference(tmp_path, monkeypatch):
    import espnet3.systems.base.backend_inference as base

    monkeypatch.setattr(base, "locate_pack", lambda t: tmp_path)
    ready = Inference(FakeSpeech2Text())
    monkeypatch.setattr(base, "load_model", lambda p, **kw: ready)
    assert Inference.from_pretrained(tmp_path) is ready

    from espnet3.api.inference import Field
    from espnet3.systems.base.backend_inference import BackendInference

    class Other(BackendInference):
        inputs = (Field("text", "text"),)
        outputs = (Field("speech", "audio"),)

        def run(self, text):
            return {}

    monkeypatch.setattr(base, "load_model", lambda p, **kw: Other(object()))
    from espnet3.api.inference import ModelTagError

    with pytest.raises(ModelTagError, match="builds Other, not Inference"):
        Inference.from_pretrained(tmp_path)


class FakeTransducer(FakeSpeech2Text):
    """The transducer Speech2Text: raw hypotheses unless asked for text."""

    def __init__(self, fs="16k", return_decoded_hyp=False, **kwargs):
        super().__init__(fs=fs, **kwargs)
        self.return_decoded_hyp = return_decoded_hyp

    def __call__(self, speech):
        if self.return_decoded_hyp:
            return super().__call__(speech)
        return [SimpleNamespace(score=0.0, yseq=[1, 2])]  # a Hypothesis

    batch_decode = None


def test_a_built_transducer_is_asked_for_decoded_text(monkeypatch):
    import espnet2.bin.asr_transducer_inference as transducer

    monkeypatch.setattr(transducer, "Speech2Text", FakeTransducer)
    path = "espnet2.bin.asr_transducer_inference.Speech2Text"
    model = Inference(backend_class=path, asr_train_config="c.yaml")
    assert model.backend.return_decoded_hyp is True
    assert model(np.zeros(16))["text"] == "hello world"
    with pytest.raises(ValueError, match="return_decoded_hyp=False"):
        Inference(backend_class=path, return_decoded_hyp=False)
    # one built elsewhere without it is named, not a bare TypeError
    with pytest.raises(TypeError, match="needs return_decoded_hyp=True"):
        Inference(FakeTransducer())(np.zeros(16))


def _enhancing(use_wpe=False, use_beamformer=False, enh_s2t=False):
    backend = FakeSpeech2Text()
    enhancer = SimpleNamespace(use_wpe=use_wpe, use_beamformer=use_beamformer)
    backend.asr_model = SimpleNamespace(frontend=SimpleNamespace(frontend=enhancer))
    backend.enh_s2t_task = enh_s2t
    return backend


def test_channels_reach_a_model_that_uses_them_as_espnet2_lays_them_out():
    stereo = np.stack([np.ones(160), -np.ones(160)]).astype(np.float32)  # (C, T)
    for backend in (
        _enhancing(use_beamformer=True),
        _enhancing(use_wpe=True),
        _enhancing(enh_s2t=True),
    ):
        model = Inference(backend)
        assert model.takes_channels
        model(Audio(stereo, 16000))
        assert backend.seen.shape == (160, 2)  # (samples, channels), as soundfile
        assert (backend.seen[:, 1] == -1).all()


def test_other_models_get_channel_zero_as_default_frontend_would_pick():
    stereo = np.stack([np.ones(160), -np.ones(160)]).astype(np.float32)
    for backend in (FakeSpeech2Text(), _enhancing()):
        model = Inference(backend)
        assert not model.takes_channels
        model(Audio(stereo, 16000))
        assert backend.seen.shape == (160,) and (backend.seen == 1).all()
    mono = FakeSpeech2Text()
    Inference(mono)(np.zeros(160, dtype=np.float32))
    assert mono.seen.ndim == 1


def test_a_batch_with_channels_for_a_model_that_uses_them_goes_one_by_one():
    backend = _enhancing(use_beamformer=True)
    stereo = np.zeros((2, 160), dtype=np.float32)
    items = [{"speech": Audio(stereo, 16000)}] * 2
    assert [o["text"] for o in Inference(backend).batch(items)] == ["hello world"] * 2
    assert not hasattr(backend, "batches")  # batch_decode reads one channel
    # a mono batch for the same model is still one beam search
    mono = [{"speech": np.zeros(160, dtype=np.float32)}] * 2
    assert [o["text"] for o in Inference(backend).batch(mono)] == ["utt0", "utt1"]
    assert backend.batches == [2]


def test_takes_channels_reads_a_real_default_frontend():
    """Count only WPE or a beamformer, not the stage DefaultFrontend always builds.

    The default stage enables neither and lets channels through to channel 0.
    """
    from espnet2.asr.frontend.default import DefaultFrontend

    for conf, uses in [
        (None, False),
        ({"use_wpe": True}, True),
        ({"use_beamformer": True}, True),
    ]:
        backend = FakeSpeech2Text()
        kwargs = {} if conf is None else {"frontend_conf": conf}
        backend.asr_model = SimpleNamespace(frontend=DefaultFrontend(**kwargs))
        assert Inference(backend).takes_channels is uses, conf


@pytest.fixture()
def tiny_asr_pack(tmp_path):
    """A pack_model-shaped bundle around a real, randomly initialised Speech2Text."""
    import string

    import yaml

    from espnet2.tasks.asr import ASRTask

    tokens = tmp_path / "tokens.txt"
    tokens.write_text(
        "\n".join(["<blank>", *string.ascii_lowercase, "<unk>", "<sos/eos>"]) + "\n"
    )
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(tokens),
            "--token_type",
            "char",
            "--decoder",
            "rnn",
        ]
    )
    pack = tmp_path / "pack"
    (pack / "conf").mkdir(parents=True)
    (pack / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "system": "esp2_asr",
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            }
        )
    )
    (pack / "conf" / "inference.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "_target_": "espnet3.systems.esp2_asr.inference.Inference",
                    "asr_train_config": str(tmp_path / "asr" / "config.yaml"),
                    "beam_size": 1,
                    "ctc_weight": 0.3,
                    "nbest": 1,
                }
            }
        )
    )
    return pack


def test_any_speech2text_argument_overrides_the_packed_one(tiny_asr_pack):
    """Several at once, decoding or not, reach the real Speech2Text."""
    from espnet3.api.inference import load

    packed = load(tiny_asr_pack)
    assert packed.backend.beam_search.beam_size == 1
    assert packed.backend.beam_search.weights["ctc"] == 0.3
    assert packed.backend.nbest == 1

    model = load(
        tiny_asr_pack,
        beam_size=3,
        ctc_weight=0.5,
        penalty=0.2,
        nbest=2,
        maxlenratio=0.5,
    )
    search = model.backend.beam_search
    assert search.beam_size == 3
    assert search.weights["ctc"] == 0.5 and search.weights["decoder"] == 0.5
    assert search.weights["length_bonus"] == 0.2
    assert model.backend.nbest == 2 and model.backend.maxlenratio == 0.5
    assert isinstance(model(np.zeros(1600, dtype=np.float32))["text"], str)

    # the bundle is unchanged: the next load reads it as packed
    assert load(tiny_asr_pack).backend.beam_search.beam_size == 1
    with pytest.raises(TypeError, match="beam_width=4: the bundle's model"):
        load(tiny_asr_pack, beam_width=4)
