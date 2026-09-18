#!/usr/bin/env python3
"""Run a published ESPnet model from the command line.

    espnet asr audio.wav
    espnet asr audio.wav --language jpn
    espnet translate audio.wav --to eng
    espnet tts "Hello from ESPnet" -o hello.wav
    espnet enhance noisy.wav -o clean.wav
    espnet models

Every espnet2.bin.*_inference module already has a command line, but it is
the one a recipe needs: an scp file, a data type triple and an output
directory. Trying a published model on one audio file therefore meant
writing Python. These subcommands take a file and print, or write, the
result.

Each downloads its model on first use and keeps it in the espnet_model_zoo
cache; `--model` takes any tag from https://huggingface.co/espnet.
"""

import argparse
import os
import sys
from typing import List, Optional

# One flagship per task, so that `espnet asr x.wav` works with no arguments.
# Each is checked by test_cli.py against the espnet2 class that loads it.
DEFAULT_MODELS = {
    "asr": "espnet/owsm_ctc_v4_1B",
    "translate": "espnet/owsm_ctc_v4_1B",
    "tts": "espnet/kan-bayashi_ljspeech_vits",
    "enhance": "espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw",
}
# OWSM writes languages as ISO 639-3 in its own token symbols.
DEFAULT_LANGUAGE = "nolang"  # let the model detect it


class CLIError(RuntimeError):
    """Something the user can fix, reported without a traceback."""


def _require_file(path: str) -> str:
    """Fail before a model is downloaded, not after."""
    if not os.path.isfile(path):
        raise CLIError(f"no such file: {path}")
    return path


def _load_audio(path: str):
    """Read an audio file as float32, at whatever rate it holds."""
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover - soundfile is a core dependency
        raise CLIError("soundfile is not installed: pip install espnet") from e
    _require_file(path)
    try:
        speech, rate = sf.read(path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as e:
        raise CLIError(f"cannot read {path}: {e}") from e
    if speech.ndim > 1:  # mix down: these models take one channel
        speech = speech.mean(axis=1)
    return speech, rate


def _write_audio(path: str, wave, rate: int) -> None:
    import soundfile as sf

    sf.write(path, wave, rate)
    print(f"wrote {path}", file=sys.stderr)


def cmd_asr(args) -> int:
    _require_file(args.audio)
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = Speech2TextGreedySearch.from_pretrained(args.model, device=args.device)
    print(s2t.batch_decode(args.audio, lang_sym=f"<{args.language}>", task_sym="<asr>"))
    return 0


def cmd_translate(args) -> int:
    _require_file(args.audio)
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = Speech2TextGreedySearch.from_pretrained(args.model, device=args.device)
    print(
        s2t.batch_decode(
            args.audio, lang_sym=f"<{args.language}>", task_sym=f"<st_{args.to}>"
        )
    )
    return 0


def cmd_tts(args) -> int:
    from espnet2.bin.tts_inference import Text2Speech

    tts = Text2Speech.from_pretrained(args.model, device=args.device)
    output = tts(args.text)
    _write_audio(args.output, output["wav"].view(-1).cpu().numpy(), tts.fs)
    return 0


def cmd_enhance(args) -> int:
    _require_file(args.audio)
    from espnet2.bin.enh_inference import SeparateSpeech

    speech, rate = _load_audio(args.audio)
    enh = SeparateSpeech.from_pretrained(args.model, device=args.device)
    waves = enh(speech[None, :], fs=rate)
    if len(waves) == 1:
        _write_audio(args.output, waves[0][0], rate)
    else:  # a separation model returns one wave per speaker
        stem, _, suffix = args.output.rpartition(".")
        for i, wave in enumerate(waves, start=1):
            _write_audio(f"{stem}.spk{i}.{suffix}", wave[0], rate)
    return 0


def cmd_models(args) -> int:
    print("Defaults, each overridable with --model <tag>:\n")
    for task, tag in DEFAULT_MODELS.items():
        print(f"  {task:10} {tag}")
    print(
        "\nEvery model in https://huggingface.co/espnet works as a tag."
        "\nThe first run downloads one; it is cached afterwards."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="espnet",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1].split("\n")),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, func):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--model", default=DEFAULT_MODELS.get(name), help="model tag")
        p.add_argument("--device", default="cpu", help="cpu, cuda, mps (default: cpu)")
        p.set_defaults(func=func)
        return p

    p = add("asr", "transcribe an audio file", cmd_asr)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="ISO 639-3 code of the speech, e.g. eng, jpn (default: detect)",
    )

    p = add("translate", "translate speech into another language", cmd_translate)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument("--to", required=True, help="ISO 639-3 target, e.g. eng, jpn")
    p.add_argument(
        "--language", default=DEFAULT_LANGUAGE, help="ISO 639-3 code of the speech"
    )

    p = add("tts", "synthesise speech from text", cmd_tts)
    p.add_argument("text", help="what to say")
    p.add_argument("-o", "--output", default="out.wav", help="output wav")

    p = add("enhance", "remove noise from an audio file", cmd_enhance)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument("-o", "--output", default="enhanced.wav", help="output wav")

    sub.add_parser(
        "models", help="show the default model of each command"
    ).set_defaults(func=cmd_models)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CLIError as e:
        print(f"espnet: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130


if __name__ == "__main__":
    sys.exit(main())
