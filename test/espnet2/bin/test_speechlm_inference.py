"""Both ways of running Bagpiper, without an 8B model on either side.

The served path is checked against a stand-in server: what matters is the
request, because the two things that decide whether a Bagpiper request
works - the constant system prompt and the temperature contract - are
carried by the payload. The loaded path is checked against a stand-in
model and preprocessor: what matters is the dialogue it builds and the
decoding config it asks for.
"""

import base64
import io
import json
import urllib.error
import wave

import numpy as np
import pytest
import soundfile as sf
import torch

from espnet2.bin import speechlm_inference as mod
from espnet2.bin.speechlm_inference import (
    TTS_SYSTEM,
    Decoding,
    LocalBagpiper,
    ServedBagpiper,
)


def wav_bytes(seconds=0.1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sound.wav"
    path.write_bytes(wav_bytes())
    return path


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch):
    """A server that records the request and answers what it is told to."""

    sent = {}

    def answer(text="an answer", audio=None):
        message = {"content": text}
        if audio is not None:
            message["audio"] = {"data": base64.b64encode(audio).decode()}
        return {"choices": [{"message": message, "finish_reason": "stop"}]}

    sent["reply"] = answer()

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(sent["reply"])

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    sent["answer"] = answer
    return sent


# --- served ------------------------------------------------------------


def test_describe_sends_the_audio_and_stops_at_eot(server, audio_file):
    text = ServedBagpiper().describe(audio_file, prompt="What is this?")

    assert text == "an answer"
    payload = server["payload"]
    assert payload["stop_token_ids"] == [mod.EOT]
    part = payload["messages"][1]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert base64.b64decode(part["input_audio"]["data"]) == audio_file.read_bytes()
    assert payload["messages"][-1]["content"] == "What is this?"


def test_describe_refuses_a_missing_file(server, tmp_path):
    with pytest.raises(FileNotFoundError):
        ServedBagpiper().describe(tmp_path / "nope.wav")


def test_render_sends_the_system_prompt_verbatim(server):
    ServedBagpiper().render("A door closing in a stairwell.")

    assert server["payload"]["messages"][0] == {
        "role": "system",
        "content": TTS_SYSTEM,
    }


def test_render_holds_the_temperature_contract(server):
    ServedBagpiper().render("Rain on a tin roof.")

    payload = server["payload"]
    extra = payload["vllm_xargs"]
    # the sampler runs at one temperature for the whole request
    assert payload["temperature"] == extra["audio_temperature"]
    assert payload["top_k"] == extra["audio_topk"]
    assert extra["text_temperature"] == 0.6
    assert extra["mode"] == "text_audio"
    # the phase machine ends the audio itself
    assert "stop_token_ids" not in payload


def test_render_sends_guidance_only_when_asked(server):
    ServedBagpiper().render("A bell.")
    assert "cfg" not in server["payload"]["vllm_xargs"]

    ServedBagpiper().render("A bell.", decoding=Decoding(cfg=3.0))
    assert server["payload"]["vllm_xargs"]["cfg"] == 3.0


def test_render_returns_the_audio_and_the_text(server):
    server["reply"] = server["answer"]("a bell rings", audio=wav_bytes())
    audio, text = ServedBagpiper().render("A bell.")

    assert text == "a bell rings"
    assert audio == wav_bytes()


def test_a_text_only_answer_is_not_a_failure(server):
    audio, text = ServedBagpiper().render("A bell.")

    assert audio is None
    assert text == "an answer"


def test_the_address_and_model_name_are_the_caller_s(server):
    ServedBagpiper(url="http://gpu-7:8000/v1/", model="bagpiper-tts").ask("hi")

    assert server["url"] == "http://gpu-7:8000/v1/chat/completions"
    assert server["payload"]["model"] == "bagpiper-tts"


def test_nothing_listening_says_how_to_start_it(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)

    with pytest.raises(RuntimeError) as raised:
        ServedBagpiper().ask("hi")

    assert "docker run" in str(raised.value) or "README" in str(raised.value)
    assert "from_pretrained" in str(raised.value)


# --- loaded ------------------------------------------------------------


PUBLISHED_TEXT = {
    "dtype": "bfloat16",
    "num_hypo": 1,
    "enforce_modality": ["text"],
    "audio": {"temperature": 0.8, "topk": 20, "cfg": 3, "max_step": 2048},
    "text": {"temperature": 0.6, "topk": 20, "cfg": 1, "max_step": 2048},
}
PUBLISHED_AUDIO = dict(PUBLISHED_TEXT, enforce_modality=["text", "audio"])


class FakePreprocessor:
    def __init__(self):
        self.seen = None

    def collate_fn(self, data_lst):
        self.seen = data_lst
        return {"keys": [data_lst[0][0]], "seqs": torch.zeros(1, 1, 1).long()}


class FakeModel:
    def __init__(self, answer):
        self.answer = answer
        self.config = None
        self.batch = None

    def inference(self, config, **batch):
        self.config = config
        self.batch = batch
        return self.answer, None


def local(answer):
    preprocessor = FakePreprocessor()
    model = FakeModel(answer)
    engine = LocalBagpiper(
        model=model,
        preprocessor=preprocessor,
        text_config=dict(PUBLISHED_TEXT),
        audio_config=dict(PUBLISHED_AUDIO),
        device="cpu",
        dtype=torch.float32,
    )
    return engine, model, preprocessor


def test_loaded_describe_builds_the_dialogue_the_dataloader_would(audio_file):
    engine, model, preprocessor = local([["assistant", "text", ["a bell"]]])

    assert engine.describe(audio_file, prompt="What is this?") == "a bell"

    ((key, data),) = preprocessor.seen
    assert key[0] == "dialogue"
    roles = [(role, modality) for role, modality, _ in data["dialogue"]]
    assert roles == [("system", "text"), ("user", "audio"), ("user", "text")]
    # audio arrives as the loader hands it over, not as a path
    array, rate = data["dialogue"][1][2]
    assert isinstance(array, np.ndarray) and array.ndim == 2
    assert rate == 16000
    assert data["dialogue"][2][2] == "What is this?"
    # the batch reached the model without the key the script pops
    assert "keys" not in model.batch


def test_loaded_asks_for_the_modalities_the_task_needs(audio_file):
    engine, model, _ = local([["assistant", "text", ["a bell"]]])

    engine.describe(audio_file)
    assert model.config["enforce_modality"] == ["text"]

    engine.render("A bell.")
    assert model.config["enforce_modality"] == ["text", "audio"]


def test_loaded_decoding_overrides_the_published_config(audio_file):
    engine, model, _ = local([["assistant", "text", ["a bell"]]])

    engine.render("A bell.", decoding=Decoding(cfg=1.0, max_steps=64))
    assert model.config["audio"]["cfg"] == 1.0
    assert model.config["audio"]["max_step"] == 64
    assert model.config["text"]["temperature"] == 0.6

    engine.render("A bell.")
    # the published default, not the last call's
    assert model.config["audio"]["cfg"] == 3


def test_loaded_render_returns_a_wav_and_the_text():
    speech = torch.zeros(1, 1, 800)
    engine, _, _ = local(
        [
            ["assistant", "text", ["a bell rings"]],
            ["assistant", "audio", (speech, torch.tensor([800]), 16000)],
        ]
    )

    audio, text = engine.render("A bell.")

    assert text == "a bell rings"
    data, rate = sf.read(io.BytesIO(audio))
    assert rate == 16000
    assert len(data) == 800


def test_loaded_text_only_answer_is_not_a_failure():
    engine, _, _ = local([["assistant", "text", ["no audio this time"]]])

    audio, text = engine.render("A bell.")

    assert audio is None
    assert text == "no audio this time"


def test_the_repeated_turn_builder_agrees_with_the_dataloader(audio_file):
    """The loaded path repeats the dataloader's conversion; check it matches."""
    loader = pytest.importorskip(
        "espnet2.speechlm.dataloader.multimodal_loader.dialogue_loader",
        reason="the speechlm data stack is not installed",
    )

    turn = ["user", "audio", str(audio_file)]
    mine = mod._dialogue_turn(tuple(turn))
    (theirs,) = loader.validate_and_process_messages([list(turn)], key="x")

    assert mine[:2] == tuple(theirs[:2])
    assert np.array_equal(mine[2][0], theirs[2][0])
    assert mine[2][1] == theirs[2][1]


# --- the three checkpoints ---------------------------------------------


def test_the_tts_checkpoint_is_sent_no_system_turn():
    """Its training data has none, so adding one is off distribution."""
    engine, model, preprocessor = local([["assistant", "text", ["ok"]]])
    engine.tts_system = mod.RELEASES["espnet/bagpiper-tts-sft"]["system"]

    engine.render("Say 'your package arrives Tuesday' in a calm voice.")

    ((_, data),) = preprocessor.seen
    assert [role for role, _, _ in data["dialogue"]] == ["user"]


def test_the_general_checkpoint_is_sent_the_constant():
    engine, _, preprocessor = local([["assistant", "text", ["ok"]]])

    engine.render("A bell.")

    ((_, data),) = preprocessor.seen
    assert data["dialogue"][0] == ("system", "text", TTS_SYSTEM)


def test_an_explicit_system_turn_wins(server):
    ServedBagpiper().render("A bell.", system=None)
    assert server["payload"]["messages"][0]["role"] == "user"


def test_a_served_tts_model_gets_no_system_turn(server):
    ServedBagpiper(model="bagpiper-tts-sft").render("A bell.")
    assert server["payload"]["messages"][0]["role"] == "user"

    ServedBagpiper(model="bagpiper").render("A bell.")
    assert server["payload"]["messages"][0]["content"] == TTS_SYSTEM


def test_the_base_checkpoint_says_it_is_not_the_assistant():
    with pytest.raises(ValueError) as raised:
        LocalBagpiper.from_pretrained("espnet/bagpiper")

    assert "espnet/bagpiper-sft" in str(raised.value)


def test_one_decoding_config_serves_both_directions(tmp_path):
    """The TTS release ships `inference.yaml` and nothing else."""
    (tmp_path / "inference.yaml").write_text(
        "dtype: bfloat16\ntext: {temperature: 0.6}\naudio: {cfg: 3}\n"
    )

    config = mod._decoding_config(tmp_path, "inference_audio.yaml")

    assert config["audio"]["cfg"] == 3


def test_no_decoding_config_falls_back_to_the_paper(tmp_path):
    config = mod._decoding_config(tmp_path, "inference_text.yaml")

    assert config["audio"]["temperature"] == 0.8
    assert config["audio"]["cfg"] == 3
    assert config["text"]["temperature"] == 0.6
    # a copy, so that a caller's Decoding cannot edit the constant
    config["audio"]["cfg"] = 99
    assert mod.PAPER_DECODING["audio"]["cfg"] == 3


def test_thinking_is_separable_from_the_answer():
    thought, rest = mod.split_thinking("<think>a bell, metallic</think>A bell.")
    assert thought == "a bell, metallic"
    assert rest == "A bell."

    assert mod.split_thinking("A bell.") == ("", "A bell.")


# --- the command line --------------------------------------------------


class FakeSpeechLM:
    """Stands in for a loaded or served model: records how it was asked."""

    def __init__(self, text="a bell rings", audio=None):
        self.text = text
        self.audio = audio
        self.described = []
        self.rendered = []

    def describe(self, audio, prompt=None, **kwargs):
        self.described.append((str(audio), prompt))
        return self.text

    def render(self, scene, decoding=None, **kwargs):
        self.rendered.append((scene, decoding))
        return self.audio, self.text


@pytest.fixture
def cli_model(monkeypatch):
    """Whatever the command builds, without a checkpoint or a server."""
    from espnet2.bin import cli

    model = FakeSpeechLM()
    asked = {}

    def from_pretrained(tag, device=None, **kwargs):
        asked["tag"], asked["device"] = tag, device
        return model

    def from_server(url, **kwargs):
        asked["url"] = url
        return model

    monkeypatch.setattr(mod, "from_pretrained", from_pretrained)
    monkeypatch.setattr(mod, "from_server", from_server)
    monkeypatch.setattr(cli, "_require_file", lambda path: None)
    model.asked = asked
    return model


def test_describe_prints_the_answer(cli_model, capsys):
    from espnet2.bin import cli

    assert cli.main(["describe", "recording.wav"]) == 0

    assert capsys.readouterr().out.strip() == "a bell rings"
    assert cli_model.asked["tag"] == cli.DEFAULT_MODELS["describe"]
    # an 8B model: the command picks the GPU rather than defaulting to cpu
    assert cli_model.asked["device"] is None
    assert cli_model.described == [("recording.wav", mod.UNDERSTAND_PROMPT)]


def test_describe_can_drop_the_reasoning(cli_model, capsys):
    from espnet2.bin import cli

    cli_model.text = "<think>metallic, two strikes</think>A bell rings twice."

    assert cli.main(["describe", "recording.wav", "--brief"]) == 0

    assert capsys.readouterr().out.strip() == "A bell rings twice."


def test_a_server_is_addressed_instead_of_a_checkpoint(cli_model, capsys):
    from espnet2.bin import cli

    assert cli.main(["describe", "r.wav", "--server", "http://gpu-7:9811/v1"]) == 0

    assert cli_model.asked == {"url": "http://gpu-7:9811/v1"}


def test_render_writes_the_wav_and_prints_the_plan(cli_model, tmp_path, capsys):
    from espnet2.bin import cli

    cli_model.audio = wav_bytes()
    out = tmp_path / "bell.wav"

    assert cli.main(["render", "A bell rings.", "-o", str(out)]) == 0

    assert out.read_bytes() == wav_bytes()
    assert capsys.readouterr().out.strip() == "a bell rings"
    assert cli_model.rendered[0][0] == "A bell rings."


def test_render_passes_guidance_when_it_is_given(cli_model, tmp_path):
    from espnet2.bin import cli

    cli_model.audio = wav_bytes()
    out = tmp_path / "bell.wav"

    cli.main(["render", "A bell.", "-o", str(out), "--cfg", "1"])
    assert cli_model.rendered[0][1].cfg == 1.0

    cli.main(["render", "A bell.", "-o", str(out)])
    # None leaves the published default in place rather than sending a 1
    assert cli_model.rendered[1][1].cfg is None


def test_render_with_no_audio_says_what_kind_of_prompt_works(
    cli_model, tmp_path, capsys
):
    from espnet2.bin import cli

    out = tmp_path / "nothing.wav"

    assert cli.main(["render", "read this aloud: hello", "-o", str(out)]) == 1

    assert not out.exists()
    assert "described scene" in capsys.readouterr().err


def test_render_reads_the_scene_from_a_pipe(cli_model, tmp_path, monkeypatch):
    from espnet2.bin import cli

    cli_model.audio = wav_bytes()
    monkeypatch.setattr("sys.stdin", io.StringIO("A bell rings twice.\n"))

    assert cli.main(["render", "-", "-o", str(tmp_path / "b.wav")]) == 0

    assert cli_model.rendered[0][0] == "A bell rings twice."
