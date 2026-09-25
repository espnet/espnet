#!/usr/bin/env python3
"""Run a SpeechLM checkpoint on one request, served or loaded.

This is the single-sample sibling of `espnet2/speechlm/bin/inference.py`,
which runs the same models over a dataset: same job template, same
checkpoints, same decoding configs, one dialogue at a time and no
manifests. Use that one to evaluate a test set, and this one to answer a
question. Everything published for this task so far is Bagpiper, so the
classes here carry its prompt contract and its name.

[Bagpiper](https://openreview.net/forum?id=FuHs64E3X6) is an 8B speech
language model that takes text and audio and answers with text or with
audio it renders. There are two ways to run it, and this module gives both
the same three methods, so that calling code does not care which one is
behind it::

    from espnet2.bin import bagpiper

    model = bagpiper.from_server()                    # the vLLM fork
    model = bagpiper.from_pretrained("espnet/bagpiper-sft")   # this package

    model.describe("recording.wav")     # what is in this audio, in words
    model.render("A cat meowing twice in an empty kitchen.")  # audio out
    model.ask("Name three percussion instruments.")            # text only

`from_server` speaks to the [ESPnet vLLM fork](https://github.com/espnet/vllm)
over its OpenAI-compatible API, which is the fast path and the one
`egs2/bagpiper/speechlm1/README.md` documents::

    docker run --rm --gpus all \\
        -v /path/to/converted-checkpoint:/models/bagpiper \\
        -v ~/.cache/huggingface:/root/.cache/huggingface \\
        -p 127.0.0.1:9811:9811 \\
        --entrypoint bash espnet/vllm:latest \\
        -c 'MODEL_PATH=/models/bagpiper bash \\
            /workspace/vllm-fork/examples/espnet/serve_bagpiper.sh'

`from_pretrained` needs no server and no fork: it loads the published
checkpoint with this package's own `espnet2.speechlm` and calls the model
in process. Every release ships exactly what that path takes - a train
config, `model.pt` as `{"module": state_dict}`, and the decoding configs
this module uses as its defaults - and each model card names
`espnet2/speechlm/bin/inference.py`, the batch script this is the
single-sample sibling of, as its runtime. These are 8B models with an
audio encoder and a codec attached: expect an 18 GB download and a large
GPU.

There are three of them, and `RELEASES` holds what a caller cannot guess:

`espnet/bagpiper-sft`
    The general assistant. Understanding and generation over speech,
    music and sound, and the default here.
`espnet/bagpiper-tts-sft`
    [Bagpiper-TTS](https://arxiv.org/abs/2606.22811): speech synthesis
    driven by a natural-language request, covering multi-talker,
    intent-to-speech, role-play and singing as well as plain reading. No
    reference audio, so it is not voice cloning. Its training data
    carries **no system turn**, so `render()` sends none for it.
`espnet/bagpiper`
    The pre-trained base, for probing and initialisation. It was never
    fine-tuned on requests, so `from_pretrained` refuses it unless asked
    twice.

Both directions of this model think before they answer: understanding is
trained as `[Audio, User Request, Rich Caption, CoT, Answer]` and
generation as `[User Request, CoT, Rich Caption, Audio]`
(arXiv:2602.05220 2.3.2). The rich caption is the model's own account of
the audio, and it is the same object in both directions - which is what
makes describe-then-render a round trip rather than two demos.
`split_thinking()` separates it out where it is marked.

The two paths express one contract twice, and the mapping is worth stating
because it is what lets one class cover both:

======================  ==========================  ======================
knob                    served                      loaded
======================  ==========================  ======================
text then audio         `mode="text_audio"`         `enforce_modality`
text only               `stop_token_ids=[3]`        `enforce_modality`
text temperature        `vllm_xargs`                `text.temperature`
audio temperature       top-level `temperature`     `audio.temperature`
guidance                `vllm_xargs["cfg"]`         `audio.cfg`
======================  ==========================  ======================

Two facts decide whether a request works at all, on either path:

**Bagpiper is not a text-to-speech engine.** It renders a described scene.
A prompt is a description with any spoken line quoted inside it; "read this
aloud: ..." is not a shape it was trained on, and returns audio that is not
a faithful reading. `render()` says so in its own docstring, and is named
apart from `espnet synthesize` for that reason.

**The audio-generation system prompt is a constant**, `TTS_SYSTEM` below,
carried by every sampled entry of the SFT data. Paraphrasing it loses the
think-then-describe-then-render behaviour that produces audio at all.

The served path imports nothing but the standard library: a caller of a
served model should not need the inference stack, and `pip install espnet`
does not carry an HTTP client either. Everything torch is imported inside
the loaded path.
"""

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_URL = os.environ.get("ESPNET_BAGPIPER_URL", "http://127.0.0.1:9811/v1")
DEFAULT_MODEL = os.environ.get("ESPNET_BAGPIPER_MODEL", "bagpiper")
DEFAULT_TAG = "espnet/bagpiper-sft"


# The one system prompt every sampled entry of the SFT data carries, verbatim.
# It is what puts the model into think-then-describe-then-render mode; the
# fork's reference client measured audio returned for 7 of 8 requests with it
# and 0 of 8 without.
TTS_SYSTEM = (
    "You are a helpful assistant that generates audio based on user requests. "
    "You can create various types of audio including sound effects, music, "
    "speech, ambient sounds, and any combination of these. When given a "
    "request, first think through what the user wants and how to create "
    "high-quality audio, then provide a detailed description of the audio you "
    "will generate."
)
UNDERSTAND_SYSTEM = "You are an audio understanding assistant."
UNDERSTAND_PROMPT = "What sound is in this audio?"

# The three published checkpoints, and the one thing about each that a caller
# cannot guess. `system` is the generation system turn the checkpoint was
# trained under: the general SFT data carries `TTS_SYSTEM` on every sampled
# entry, and the TTS data carries none at all - "Our data does not include any
# system prompt, so the exact pre-defined application is agnostic to the
# model" (Bagpiper-TTS, arXiv:2606.22811 2.3). Sending one anyway is off
# distribution.
RELEASES = {
    "espnet/bagpiper-sft": {
        "what": "understanding and generation over speech, music and sound",
        "system": TTS_SYSTEM,
    },
    "espnet/bagpiper-tts-sft": {
        "what": "natural-language-guided speech synthesis, no reference audio",
        "system": None,
    },
    "espnet/bagpiper": {
        "what": "the pre-trained base, for probing and initialisation",
        "system": None,
    },
}

NOT_AN_ASSISTANT = (
    "{tag} is the pre-trained base checkpoint, not the instruction-tuned "
    "assistant: it was never fine-tuned on requests, so describe()/render()/"
    "ask() are off distribution. Its own model card sends you to "
    "espnet/bagpiper-sft. Pass allow_base=True to load it anyway."
)

# Table 10 of arXiv:2602.05220, which is also what all three published
# decoding configs hold. The base checkpoint ships none, so this is what it
# falls back to.
PAPER_DECODING: Dict[str, Any] = {
    "dtype": "bfloat16",
    "num_hypo": 1,
    "text": {"temperature": 0.6, "topk": 20, "cfg": 1, "max_step": 2048, "min_step": 1},
    "audio": {
        "temperature": 0.8,
        "topk": 20,
        "cfg": 3,
        "max_step": 2048,
        "min_step": 50,
    },
}

# `<|eot|>`, which ends a text answer. The audio tasks send no stop token: the
# server's phase machine ends those itself.
EOT = 3

NOT_SERVING = (
    "no Bagpiper server at {url}. It is served rather than loaded - start it "
    "with the docker run in egs2/bagpiper/speechlm1/README.md, pass the "
    "address of one that is already running, or load the checkpoint in "
    "process with espnet2.bin.bagpiper.from_pretrained()."
)

#: "whatever this checkpoint was trained under", as a default argument that
#: None has to stay distinguishable from.
AUTO = "<the checkpoint's own>"

# One turn, before either path renders it: who says it, in what modality,
# and either the text or the path to the audio file.
Message = Tuple[str, str, str]

logger = logging.getLogger(__name__)


@dataclass
class Decoding:
    """How to sample, in the terms both paths can be given.

    The defaults are the published ones: `inference_text.yaml` and
    `inference_audio.yaml` of `espnet/bagpiper-sft`, which the fork's
    reference client also sends.

    Args:
        text_temperature: For the text the model writes.
        audio_temperature: For the codec frames it renders.
        topk: Top-k for both.
        cfg: Classifier-free guidance for the audio. None keeps each
            path's own default (the published 3, or nothing on the wire);
            1.0 is equivalent to none, and anything above it doubles the
            KV cache one request needs.
        max_tokens: Ceiling on a served request.
        max_steps: Ceiling on one segment of a loaded one.
    """

    text_temperature: float = 0.6
    audio_temperature: float = 0.8
    topk: int = 20
    cfg: Optional[float] = None
    max_tokens: Optional[int] = None
    max_steps: Optional[int] = None

    def tokens(self, audio: bool) -> int:
        """The served ceiling, which audio needs far more of."""
        if self.max_tokens is not None:
            return self.max_tokens
        return 12000 if audio else 4096


class Bagpiper:
    """What Bagpiper does, with where it runs left to a subclass.

    Use `from_server()` or `from_pretrained()` rather than this class.
    """

    def describe(
        self,
        audio: Union[str, Path],
        prompt: str = UNDERSTAND_PROMPT,
        system: Optional[str] = UNDERSTAND_SYSTEM,
        decoding: Optional[Decoding] = None,
    ) -> str:
        """Say in words what is in a recording.

        Args:
            audio: A local audio file.
            prompt: What to ask about it.
            system: The system turn, or None for none.
            decoding: How to sample; the published defaults if None.

        Returns:
            The answer, as text.
        """
        path = Path(audio)
        if not path.is_file():
            raise FileNotFoundError(f"no such audio file: {path}")
        messages: List[Message] = []
        if system:
            messages.append(("system", "text", system))
        messages.append(("user", "audio", str(path)))
        messages.append(("user", "text", prompt))
        text, _ = self._chat(messages, False, decoding or Decoding())
        return text

    #: The generation system turn this checkpoint was trained under, which
    #: `render()` sends unless the caller says otherwise. See `RELEASES`.
    tts_system: Optional[str] = TTS_SYSTEM

    def render(
        self,
        scene: str,
        system: Optional[str] = AUTO,
        decoding: Optional[Decoding] = None,
    ) -> Tuple[Optional[bytes], str]:
        """Render audio for a described scene.

        **Describe, do not instruct.** This model was trained on scene
        descriptions with any spoken line quoted inside them:

            "A clear, friendly female voice, close-miked in a quiet room,
            says: 'Hello, how are you today?'. She speaks at a relaxed,
            natural pace with a warm tone and no background noise."

        "Read this aloud: Hello, how are you today?" is not that shape. It
        returns audio, loosely conditioned, and not a faithful reading -
        which is why this method is `render` and not `synthesize`.

        Args:
            scene: The description, with any speech quoted inside it.
            system: The system turn. The default is whatever this
                checkpoint was trained under - the constant for the general
                SFT model, and none at all for the TTS one, which was
                trained without a system turn. Paraphrasing the constant
                costs the audio; adding one where there was none is off
                distribution the other way. None sends none.
            decoding: How to sample; the published defaults if None.

        Returns:
            The WAV bytes and the text the model wrote on the way there.
            **The audio may be None**: the model chooses its own output
            mode, and a request that stops after the text has not failed.
        """
        if system is AUTO:
            system = self.tts_system
        messages: List[Message] = []
        if system:
            messages.append(("system", "text", system))
        messages.append(("user", "text", scene))
        text, audio = self._chat(messages, True, decoding or Decoding())
        return audio, text

    def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        decoding: Optional[Decoding] = None,
    ) -> str:
        """Answer a question with no audio on either side."""
        messages: List[Message] = []
        if system:
            messages.append(("system", "text", system))
        messages.append(("user", "text", prompt))
        text, _ = self._chat(messages, False, decoding or Decoding())
        return text

    def _chat(
        self,
        messages: Sequence[Message],
        audio_out: bool,
        decoding: Decoding,
    ) -> Tuple[str, Optional[bytes]]:
        """One exchange: the text of the answer, and its audio if any."""
        raise NotImplementedError


class ServedBagpiper(Bagpiper):
    """A Bagpiper server, addressed rather than loaded.

    Args:
        url: Where the server is, up to and including `/v1`.
        model: The name the server serves it under.
        timeout: Seconds to wait for one request. Audio generation on a
            busy server is minutes rather than seconds, which is why this
            is not the usual half-minute.
        tts_system: The generation system turn, which differs by
            checkpoint (see `RELEASES`). The default reads the served
            model's name: a name with "tts" in it is taken to be
            Bagpiper-TTS, which was trained without a system turn. Pass
            one, or None, to decide it yourself.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 3600.0,
        tts_system: Optional[str] = AUTO,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # a served model is a name rather than a release, so the only thing
        # to go on is that name; anything explicit wins, including None
        if tts_system is AUTO:
            tts_system = None if "tts" in model.lower() else TTS_SYSTEM
        self.tts_system = tts_system

    def __repr__(self) -> str:
        return f"ServedBagpiper(url={self.url!r}, model={self.model!r})"

    def _chat(self, messages, audio_out, decoding):
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [_openai_turn(m) for m in messages],
            "max_tokens": decoding.tokens(audio_out),
            "top_k": decoding.topk,
        }
        if audio_out:
            extra = {
                "mode": "text_audio",
                "phase": "text",
                "text_temperature": decoding.text_temperature,
                "audio_temperature": decoding.audio_temperature,
                "audio_topk": decoding.topk,
            }
            if decoding.cfg is not None:
                extra["cfg"] = decoding.cfg
            # The sampler runs at one temperature for the whole request and
            # the model pre-multiplies the text-phase logits, so this has to
            # be the audio one for the text one to take effect. No stop
            # token either: the phase machine ends the audio segment.
            payload["temperature"] = decoding.audio_temperature
            payload["vllm_xargs"] = extra
        else:
            payload["temperature"] = decoding.text_temperature
            payload["stop_token_ids"] = [EOT]

        answer = self._post(payload)
        return _answer_text(answer), _answer_audio(answer)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One request, or a sentence saying what to do about it."""
        request = urllib.request.Request(
            f"{self.url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"{self.url}: HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            # the usual case, and the one worth a sentence: nothing is
            # listening, because nobody started the server
            raise RuntimeError(NOT_SERVING.format(url=self.url)) from e


class LocalBagpiper(Bagpiper):
    """A Bagpiper checkpoint, loaded here.

    This is the single-sample sibling of `espnet2/speechlm/bin/inference.py`:
    same job template, same checkpoint, same decoding configs, one dialogue
    at a time and no manifests. Build it with `from_pretrained()`.

    Args:
        model: The built and loaded `espnet2.speechlm` model.
        preprocessor: Its job template's preprocessor.
        text_config: Decoding config for a text answer.
        audio_config: Decoding config for a rendered one.
        device: Where the model is.
        dtype: What the batch is cast to, matching the model.
        tts_system: The generation system turn this checkpoint was
            trained under. See `RELEASES`.
    """

    def __init__(
        self,
        model,
        preprocessor,
        text_config: Dict[str, Any],
        audio_config: Dict[str, Any],
        device: str,
        dtype,
        tts_system: Optional[str] = TTS_SYSTEM,
    ):
        self.tts_system = tts_system
        self.model = model
        self.preprocessor = preprocessor
        self.text_config = text_config
        self.audio_config = audio_config
        self.device = device
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"LocalBagpiper(device={self.device!r}, dtype={self.dtype})"

    @classmethod
    def from_pretrained(
        cls,
        tag_or_dir: Union[str, Path] = DEFAULT_TAG,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        train_config: Optional[Union[str, Path]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        allow_base: bool = False,
    ) -> "LocalBagpiper":
        """Load a published checkpoint, downloading it if it is a tag.

        Args:
            tag_or_dir: A Hub tag, or a directory holding an already
                downloaded release. `RELEASES` says what each published
                one is for; `espnet/bagpiper-sft` is 18 GB, and so are
                the others.
            device: Where to put it. Defaults to cuda where there is one;
                an 8B model with an encoder and a codec attached does not
                fit anywhere smaller in practice.
            dtype: Overrides the dtype the decoding config asks for.
            train_config: The train YAML, if the directory holds more than
                one or it lives elsewhere.
            checkpoint: The `.pt`, likewise.
            allow_base: Load the pre-trained base checkpoint even though
                it was never fine-tuned to answer requests.

        Returns:
            A LocalBagpiper with the model already loaded and in eval mode.
        """
        # before the model stack is imported and before 18 GB is fetched
        release = RELEASES.get(str(tag_or_dir), {})
        if str(tag_or_dir) == "espnet/bagpiper" and not allow_base:
            raise ValueError(NOT_AN_ASSISTANT.format(tag=tag_or_dir))

        import torch
        import yaml

        from espnet2.speechlm.bin.inference import load_checkpoint
        from espnet2.speechlm.model import _all_job_types

        directory = _resolve(tag_or_dir)
        train_path = Path(train_config) if train_config else _one_yaml(directory)
        ckpt_path = Path(checkpoint) if checkpoint else _one_checkpoint(directory)

        with open(train_path) as f:
            config = yaml.safe_load(f)
        # sft ships one config per direction; tts-sft ships the single
        # text-then-audio one, since that is all it does; the base ships
        # none, and falls back to the paper's Table 10
        text_config = _decoding_config(directory, "inference_text.yaml")
        audio_config = _decoding_config(directory, "inference_audio.yaml")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype or text_config.get("dtype", "bfloat16"))

        template = _all_job_types[config["job_type"]](config, is_train=False)
        logger.info("Building %s from %s", config["job_type"], train_path)
        model = template.build_model()
        logger.info("Loading weights from %s", ckpt_path)
        model = load_checkpoint(model, ckpt_path)
        model.prepare_inference()
        model = model.to(device=device, dtype=torch_dtype).eval()

        return cls(
            model=model,
            preprocessor=template.build_preprocessor(),
            text_config=text_config,
            audio_config=audio_config,
            device=device,
            dtype=torch_dtype,
            tts_system=release.get("system", TTS_SYSTEM),
        )

    def _chat(self, messages, audio_out, decoding):
        from espnet2.speechlm.utils.data import to_device

        dialogue = [_dialogue_turn(message) for message in messages]
        batch = self.preprocessor.collate_fn(
            [(("dialogue", "local", "bagpiper"), {"dialogue": dialogue})]
        )
        batch = to_device(batch, self.device, dtype=self.dtype)
        batch.pop("keys")

        config = self._config(audio_out, decoding)
        answer, _ = self.model.inference(config, **batch)
        return _decoded(answer)

    def _config(self, audio_out: bool, decoding: Decoding) -> Dict[str, Any]:
        """The published config, with anything the caller asked for on top."""
        base = self.audio_config if audio_out else self.text_config
        config = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in base.items()
        }
        config["enforce_modality"] = ["text", "audio"] if audio_out else ["text"]
        config["text"]["temperature"] = decoding.text_temperature
        config["text"]["topk"] = decoding.topk
        config["audio"]["temperature"] = decoding.audio_temperature
        config["audio"]["topk"] = decoding.topk
        if decoding.cfg is not None:
            config["audio"]["cfg"] = decoding.cfg
        if decoding.max_steps is not None:
            config["text"]["max_step"] = decoding.max_steps
            config["audio"]["max_step"] = decoding.max_steps
        return config


def from_server(
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 3600.0,
    tts_system: Optional[str] = AUTO,
) -> ServedBagpiper:
    """Address a running server. See `ServedBagpiper`."""
    return ServedBagpiper(url=url, model=model, timeout=timeout, tts_system=tts_system)


def from_pretrained(
    tag_or_dir: Union[str, Path] = DEFAULT_TAG, **kwargs
) -> LocalBagpiper:
    """Load a checkpoint here. See `LocalBagpiper.from_pretrained`."""
    return LocalBagpiper.from_pretrained(tag_or_dir, **kwargs)


# --- the served wire ---------------------------------------------------


def _openai_turn(message: Message) -> Dict[str, Any]:
    """One turn as the server takes it: audio is a base64 content part."""
    role, modality, content = message
    if modality == "text":
        return {"role": role, "content": content}
    path = Path(content)
    return {
        "role": role,
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
                    "format": path.suffix.lstrip(".").lower(),
                },
            }
        ],
    }


def _answer_text(answer: Dict[str, Any]) -> str:
    """The text of the first choice."""
    return answer["choices"][0]["message"].get("content") or ""


def _answer_audio(answer: Dict[str, Any]) -> Optional[bytes]:
    """The WAV of the first choice, checked for being one, or None."""
    payload = answer["choices"][0]["message"].get("audio") or {}
    if not payload.get("data"):
        return None
    return _checked_wav(base64.b64decode(payload["data"]))


# --- the loaded wire ---------------------------------------------------


def _dialogue_turn(message: Message) -> Tuple[str, str, Any]:
    """One turn as the model takes it: audio read into (array, rate).

    This is what `validate_and_process_messages` in
    `espnet2/speechlm/dataloader/multimodal_loader/dialogue_loader.py` does
    to a manifest's dialogue, for the one turn at a time this module has.
    It is repeated rather than imported because that package pulls the data
    stack - pyarrow, lhotse - which a single file does not need;
    `test_speechlm_inference.py` checks the two agree wherever both can run.
    """
    import numpy as np
    import soundfile as sf

    role, modality, content = message
    if modality == "text":
        return (role, modality, content)
    audio, rate = sf.read(content, dtype="float32")
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    elif audio.ndim == 2:
        audio = audio.T  # [samples, channels] -> [channels, samples]
    else:
        raise ValueError(f"unexpected audio shape in {content}: {audio.shape}")
    return (role, modality, (audio, rate))


def _decoded(answer: Sequence[Sequence[Any]]) -> Tuple[str, Optional[bytes]]:
    """Assistant messages as the served path returns them: text, then audio.

    `model.inference` hands back `[role, modality, content]` per segment,
    text as a batch of strings and audio as `(wav, length, sample_rate)`.
    """
    import soundfile as sf

    texts: List[str] = []
    audio: Optional[bytes] = None
    for _, modality, content in answer:
        if modality == "text":
            texts.append(content[0] if not isinstance(content, str) else content)
            continue
        if audio is not None:
            # every published config asks for one audio segment; a model
            # that rendered more has said something this API cannot return
            logger.warning("keeping the first of several audio segments")
            continue
        wav, length, rate = content
        wav = wav[0][:, : length[0]] if wav[0].ndim == 2 else wav[0]
        buffer = io.BytesIO()
        sf.write(buffer, wav.cpu().float().numpy().T, int(rate), format="WAV")
        audio = _checked_wav(buffer.getvalue())
    return "\n".join(t for t in texts if t), audio


# --- shared ------------------------------------------------------------


def split_thinking(text: str) -> Tuple[str, str]:
    """Separate the model's reasoning from what it wrote afterwards.

    Both directions of this model think first: understanding is trained as
    `[Audio, User Request, Rich Caption, CoT, Answer]` and generation as
    `[User Request, CoT, Rich Caption, Audio]` (arXiv:2602.05220 2.3.2), so
    the text either method returns is the caption and the reasoning as well
    as the answer. In the served generation flow that block is wrapped in
    `<think>`, which is what this splits on.

    Args:
        text: Whatever `describe()`, `render()` or `ask()` returned.

    Returns:
        The reasoning and the rest. A text with no marker in it splits as
        `("", text)`: this is a convenience for showing the two apart, not
        a parser to rely on.
    """
    _, marker, rest = text.partition("</think>")
    if not marker:
        return "", text
    thought = text[: text.index(marker)]
    return thought.replace("<think>", "").strip(), rest.strip()


def _checked_wav(wav: bytes) -> bytes:
    """Bytes that are a WAV, so that a caller's player is not the one to say."""
    with wave.open(io.BytesIO(wav), "rb"):
        pass
    return wav


def _resolve(tag_or_dir: Union[str, Path]) -> Path:
    """A local release directory, downloading the tag if it is not one."""
    directory = Path(tag_or_dir)
    if directory.is_dir():
        return directory

    from huggingface_hub import snapshot_download

    logger.info("Downloading %s; the checkpoint alone is 18 GB", tag_or_dir)
    return Path(snapshot_download(repo_id=str(tag_or_dir)))


def _one_yaml(directory: Path) -> Path:
    """The release's train config, which it ships exactly one of."""
    found = sorted(
        p for p in directory.glob("*.yaml") if not p.name.startswith("inference")
    )
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one train config in {directory}, found "
            f"{[p.name for p in found]}; pass train_config="
        )
    return found[0]


def _one_checkpoint(directory: Path) -> Path:
    """The release's weights, likewise."""
    found = sorted(directory.glob("*.pt"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one checkpoint in {directory}, found "
            f"{[p.name for p in found]}; pass checkpoint="
        )
    return found[0]


def _decoding_config(directory: Path, name: str) -> Dict[str, Any]:
    """A release's decoding config, however many of them it ships.

    `espnet/bagpiper-sft` ships one per direction, `espnet/bagpiper-tts-sft`
    ships the single `inference.yaml` for the only direction it has, and
    `espnet/bagpiper` ships none. The three files that do exist are
    identical apart from `enforce_modality`, which `_config()` sets from
    the task anyway, so any of them is a usable base for either direction.
    """
    import yaml

    for candidate in (directory / name, directory / "inference.yaml"):
        if candidate.is_file():
            with open(candidate) as f:
                return yaml.safe_load(f)

    logger.info("%s ships no decoding config; using the paper's Table 10", directory)
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in PAPER_DECODING.items()
    }
