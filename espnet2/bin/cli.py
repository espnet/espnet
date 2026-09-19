#!/usr/bin/env python3
"""Run a published ESPnet model from the command line.

    espnet asr audio.wav
    espnet asr audio.wav --language jpn
    espnet translate audio.wav --to eng
    espnet tts "Hello from ESPnet" -o hello.wav
    espnet enhance noisy.wav -o clean.wav
    espnet demo
    espnet models

Every espnet2.bin.*_inference module already has a command line, but it is
the one a recipe needs: an scp file, a data type triple and an output
directory. Trying a published model on one audio file therefore meant
writing Python. These subcommands take a file and print, or write, the
result.

Each downloads its model on first use and keeps it in the espnet_model_zoo
cache. `--model` takes any tag from https://huggingface.co/espnet that suits
the command: a command loads one task's inference class, so a TTS tag given
to `espnet asr` is reported rather than half-loaded.

`espnet demo` is the same OWSM model in a browser instead: it serves the app
the Hugging Face Space runs, locally, and prints the URL to open.
"""

import argparse
import importlib.metadata
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from espnet2.utils.pretrained import ModelTagError, build_pretrained

# One flagship per task, so that `espnet asr x.wav` works with no arguments.
# Each is checked by test_cli.py against the espnet2 class that loads it.
DEFAULT_MODELS = {
    "asr": "espnet/owsm_ctc_v4_1B",
    "translate": "espnet/owsm_ctc_v4_1B",
    "tts": "espnet/kan-bayashi_ljspeech_vits",
    "enhance": "espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw",
    # the browser demo runs the model `espnet asr` runs, so that the two agree
    "demo": "espnet/owsm_ctc_v4_1B",
}
# OWSM writes languages as ISO 639-3 in its own token symbols.
# OWSM's own symbol for "work out the language yourself". asr and translate
# load OWSM-CTC, whose token list holds one <iso-639-3> symbol per language;
# another s2t model reached through --model may spell these differently.
DEFAULT_LANGUAGE = "nolang"


class CLIError(RuntimeError):
    """Something the user can fix, reported without a traceback."""


def _version() -> str:
    """The installed version, so that a bug report can name one."""
    try:
        return importlib.metadata.version("espnet")
    except importlib.metadata.PackageNotFoundError:
        # a source tree that was never installed: the command still runs
        return "unknown (running from a source tree)"


def _device(value: str) -> str:
    """Accept what torch accepts, so a typo fails now rather than after a download."""
    if value in ("cpu", "mps", "cuda") or re.fullmatch(r"cuda:\d+", value):
        return value
    raise argparse.ArgumentTypeError(
        f"unknown device {value!r}: use cpu, mps, cuda or cuda:<n>"
    )


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
    except (OSError, RuntimeError, TypeError) as e:
        raise CLIError(f"cannot read {path}: {e}") from e
    # channels are kept: SeparateSpeech takes (Batch, Nsamples [, Channels])
    # and a beamformer is worthless without them
    return speech, rate


def _write_audio(path: str, wave, rate: int) -> None:
    import soundfile as sf

    try:
        sf.write(path, wave, rate)
    except (OSError, RuntimeError) as e:
        # a missing directory or a read-only one is the user's to fix, so it
        # reads like the other errors rather than as a traceback
        raise CLIError(f"cannot write {path}: {e}") from e
    print(f"wrote {path}", file=sys.stderr)


def _build(loader, args, task: str):
    """Load a published model, or say why this tag cannot serve this command.

    The loading and the "wrong task" message are shared with ``espnet.load``,
    so that both front ends explain a mismatched tag the same way; only the
    wording of the fix is this command line's own.
    """
    return build_pretrained(
        loader,
        args.model,
        args.device,
        f"`espnet {task}`",
        f"Pass --model with a {task} model; `espnet models` names the default.",
    )


def cmd_asr(args) -> int:
    # every check the user can fail comes before the import: loading the s2t
    # stack takes seconds, and "no such file" should not wait for it
    if args.live or args.stream:
        return _transcribe_as_it_arrives(args)

    if not args.audio:
        # the argument is optional only because --live has nothing to name
        raise CLIError("give an audio file, or --live to record one")
    _require_file(args.audio)

    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = _build(Speech2TextGreedySearch, args, "asr")
    print(s2t.batch_decode(args.audio, lang_sym=f"<{args.language}>", task_sym="<asr>"))
    return 0


def _transcribe_as_it_arrives(args) -> int:
    """`--live` from the microphone, `--stream` from a file, same decoding."""
    if args.live and args.audio:
        raise CLIError("--live records from the microphone; do not also name a file")
    if args.stream and not args.audio:
        raise CLIError("--stream needs an audio file; --live reads the microphone")
    if args.stream:
        _require_file(args.audio)

    from espnet2.bin import live
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = _build(Speech2TextGreedySearch, args, "asr")
    try:
        source = live.from_microphone() if args.live else live.from_file(args.audio)

        def decode(chunk):
            results = s2t(chunk, lang_sym=f"<{args.language}>", task_sym="<asr>")
            return results[0][3] if results else ""

        return live.transcribe(decode, source)
    except live.LiveError as e:
        raise CLIError(str(e)) from e


def cmd_translate(args) -> int:
    _require_file(args.audio)
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = _build(Speech2TextGreedySearch, args, "translate")
    print(
        s2t.batch_decode(
            args.audio, lang_sym=f"<{args.language}>", task_sym=f"<st_{args.to}>"
        )
    )
    return 0


def _output_path(value: str) -> Path:
    """The file to write, with an extension soundfile can turn into a format.

    Checked here rather than at the write: a model is downloaded and run in
    between, and `-o notes.txt` ended in a traceback out of soundfile after
    all of that work.
    """
    path = Path(value)
    if not path.suffix:
        raise CLIError(f"output needs a file extension, e.g. {value}.wav")

    import soundfile as sf

    if path.suffix.lstrip(".").upper() not in sf.available_formats():
        known = ", ".join(sorted(f".{fmt.lower()}" for fmt in sf.available_formats()))
        raise CLIError(f"cannot write {path.suffix} audio; soundfile writes {known}")
    return path


def cmd_tts(args) -> int:
    path = _output_path(args.output)
    from espnet2.bin.tts_inference import Text2Speech

    tts = _build(Text2Speech, args, "tts")
    output = tts(args.text)
    _write_audio(str(path), output["wav"].view(-1).cpu().numpy(), tts.fs)
    return 0


def cmd_enhance(args) -> int:
    _require_file(args.audio)
    output = _output_path(args.output)
    from espnet2.bin.enh_inference import SeparateSpeech

    speech, rate = _load_audio(args.audio)
    enh = _build(SeparateSpeech, args, "enhance")
    waves = enh(speech[None, ...], fs=rate)
    if len(waves) == 1:
        _write_audio(str(output), waves[0][0], rate)
    else:  # a separation model returns one wave per speaker
        for i, wave in enumerate(waves, start=1):
            speaker = output.with_name(f"{output.stem}.spk{i}{output.suffix}")
            _write_audio(str(speaker), wave[0], rate)
    return 0


def _require_s2t(model_tag: str) -> None:
    """Stop before the download when the tag is not a speech-to-text model.

    `espnet demo` is the generic name of the command; what it serves today is
    one app, the OWSM one in `espnet2/bin/demo.py`, whose menus are the
    language and translation symbols of an OWSM token list. A tag for another
    task will not grow those menus - it will fail somewhere inside the
    constructor, after four gigabytes have been fetched - so the task is
    settled here, from the model's own Hugging Face metadata.

    A model whose metadata says nothing is let through rather than refused:
    the check exists to turn a knowable mistake into a sentence, not to
    become a second gate a valid checkpoint has to pass.
    """
    import espnet

    try:
        task = espnet._infer_task(model_tag)
    except Exception:  # unreachable Hub, no metadata, an unknown label
        return
    if task != "s2t":
        raise CLIError(
            f"`espnet demo` serves speech-to-text models, and {model_tag} is "
            f"a {task} model. Pass --model with an OWSM tag; `espnet models` "
            "names the default."
        )


def cmd_demo(args) -> int:
    """Serve the model in a browser, the way its Hugging Face Space does."""
    from espnet2.bin import demo

    # gradio is not part of `pip install espnet`, and a web framework is a
    # large thing to install by accident, so this reads like a missing file
    # rather than like a bug in the command.
    if demo.load_gradio() is None:
        raise CLIError(demo.GRADIO_MISSING)

    # The demo decides for itself unless asked, because it is the one command
    # that runs a 1B model interactively: a CPU default would be unusable on a
    # machine that has a GPU sitting idle.
    args.device = args.device or demo.default_device()

    # Before the download, not after it: the checkpoint is 4 GB and the
    # answer to "can this demo serve it" is one metadata request away.
    _require_s2t(args.model)

    # only now: importing the inference stack costs seconds, and a missing
    # package or an unusable --device should be reported instantly
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    s2t = _build(Speech2TextGreedySearch, args, "demo")
    app = demo.build_app(s2t, device=args.device, model_tag=args.model)
    url = f"http://127.0.0.1:{args.port}"
    # printed before launching: gradio's own banner goes to stdout only after
    # the server is up, and launch() then blocks until Ctrl-C
    print(f"{args.model} on {args.device}: open {url}")
    app.launch(server_port=args.port, share=args.share)
    return 0


def cmd_models(args) -> int:
    print("Defaults, each overridable with --model <tag>:\n")
    for task, tag in DEFAULT_MODELS.items():
        print(f"  {task:10} {tag}")
    print(
        "\nAny tag from https://huggingface.co/espnet that suits the command"
        "\nworks with --model. The first run downloads it; it is cached after."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="espnet",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1].split("\n")),
    )
    parser.add_argument("--version", action="version", version=f"espnet {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, func, device="cpu"):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--model",
            default=DEFAULT_MODELS.get(name),
            help=f"tag of a model for this task (default: {DEFAULT_MODELS.get(name)})",
        )
        p.add_argument(
            "--device",
            default=device,
            type=_device,
            # None means the command picks, which only `espnet demo` does
            help="cpu, mps, cuda or cuda:<n> (default: "
            + (device or "cuda when torch sees one, else cpu")
            + ")",
        )
        p.set_defaults(func=func)
        return p

    p = add("asr", "transcribe an audio file", cmd_asr)
    p.add_argument("audio", nargs="?", help="audio file, any format soundfile reads")
    p.add_argument(
        "--stream",
        action="store_true",
        help="print each window of the file as it is decoded",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="transcribe from the microphone until Ctrl-C (needs sounddevice)",
    )
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="OWSM language token, ISO 639-3: eng, jpn … (default: OWSM detects it)",
    )

    p = add("translate", "translate speech into another language", cmd_translate)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument("--to", required=True, help="OWSM target language token, ISO 639-3")
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="OWSM language token of the speech, ISO 639-3 (default: detected)",
    )

    p = add("tts", "synthesise speech from text", cmd_tts)
    p.add_argument("text", help="what to say")
    p.add_argument("-o", "--output", default="out.wav", help="output wav")

    p = add("enhance", "remove noise from an audio file", cmd_enhance)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument("-o", "--output", default="enhanced.wav", help="output wav")

    p = add("demo", "serve a model in the browser", cmd_demo, device=None)
    p.add_argument(
        "--port", type=int, default=7860, help="port to serve on (default: 7860)"
    )
    p.add_argument(
        "--share",
        action="store_true",
        help="also publish a temporary public gradio.live link",
    )

    sub.add_parser(
        "models", help="show the default model of each command"
    ).set_defaults(func=cmd_models)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CLIError, ModelTagError) as e:
        print(f"espnet: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130


if __name__ == "__main__":
    sys.exit(main())
