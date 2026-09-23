#!/usr/bin/env python3
"""Run a published ESPnet model from the command line.

    espnet transcribe audio.wav
    espnet transcribe audio.wav --language jpn
    espnet phonemize audio.wav
    espnet align audio.wav --text "the words that were said"
    espnet translate audio.wav --to eng
    espnet synthesize "Hello from ESPnet" -o hello.wav
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
to `espnet transcribe` is reported rather than half-loaded.

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

# One flagship per task, so that `espnet transcribe x.wav` works with no
# arguments.
# Each is checked by test_cli.py against the espnet2 class that loads it.
DEFAULT_MODELS = {
    "transcribe": "espnet/owsm_ctc_v4_1B",
    # POWSM, a phonetic model built on OWSM: the CTC one, which is the
    # faster of the two and the one that can read a recording of any length
    "phonemize": "espnet/powsm_ctc",
    # alignment reads the CTC head, so the model that transcribes is the
    # model that aligns
    "align": "espnet/owsm_ctc_v4_1B",
    "translate": "espnet/owsm_ctc_v4_1B",
    "synthesize": "espnet/kan-bayashi_ljspeech_vits",
    "enhance": "espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw",
    # the browser demo runs the model `espnet transcribe` runs, so the two agree
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


def _decode(s2t, audio: str, lang_sym: str, task_sym: str) -> str:
    """One recording of any length, as one line of text.

    ``decode_long`` returns ``(start, end, text)`` per segment and reads the
    checkpoint to decide how to cut the recording up: a CTC-only model in
    overlapping buffers with no search, an encoder-decoder model segment by
    segment on its own timestamps. This command prints a transcript, so the
    segments are joined.
    """
    return " ".join(
        text
        for _, _, text in s2t.decode_long(audio, lang_sym=lang_sym, task_sym=task_sym)
    )


def cmd_transcribe(args) -> int:
    # every check the user can fail comes before the import: loading the s2t
    # stack takes seconds, and "no such file" should not wait for it
    if args.live or args.stream:
        return _transcribe_as_it_arrives(args)

    if not args.audio:
        # the argument is optional only because --live has nothing to name
        raise CLIError("give an audio file, or --live to record one")
    _require_file(args.audio)

    from espnet2.bin.s2t_inference import Speech2Text

    s2t = _build(Speech2Text, args, "transcribe")
    print(_decode(s2t, args.audio, f"<{args.language}>", "<asr>"))
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
    from espnet2.bin.s2t_inference import Speech2Text

    s2t = _build(Speech2Text, args, "transcribe")
    try:
        source = live.from_microphone() if args.live else live.from_file(args.audio)

        def decode(chunk):
            # best_path, not the object itself: a window has to be decoded
            # before the next one arrives, and a search on the CTC head is
            # nowhere near that fast. This is the one place in the command
            # line where the difference is the difference between working
            # and not.
            results = s2t.best_path(
                chunk, lang_sym=f"<{args.language}>", task_sym="<asr>"
            )
            return results[0][3] if results else ""

        return live.transcribe(decode, source)
    except live.LiveError as e:
        raise CLIError(str(e)) from e


# POWSM writes each phone between slashes, so that a phone spelled like a BPE
# token is still one token: /pʰ//ɔ//s//ə//m/ is five phones, not a string to
# be read character by character.
PHONE = re.compile(r"/([^/]+)/")


def phones(decoded: str, spaced: bool = False) -> str:
    """The phones of a decoded line, without the slashes that delimit them.

    Returned as IPA - `pʰɔsəm` - or one phone at a time when asked, which is
    what anything counting or aligning them wants. Text with no slashes in it
    is passed through: a checkpoint that does not write phones this way has
    still said something, and swallowing it would be worse than printing it.
    """
    found = PHONE.findall(decoded)
    if not found:
        return decoded
    return " ".join(found) if spaced else "".join(found)


def _no_language(s2t) -> str:
    """The checkpoint's own symbol for an unknown language, or a fixable error.

    The lookup is the model's own - both POWSM checkpoints use `<unk>` and
    OWSM uses `<nolang>`, and one of the three does not record which - and
    only the wording of the fix belongs to this command line.
    """
    try:
        return s2t.no_language()
    except ValueError as e:
        raise CLIError(f"{e}; pass --language, as ISO 639-3") from e


def cmd_phonemize(args) -> int:
    _require_file(args.audio)
    from espnet2.bin.s2t_inference import Speech2Text

    s2t = _build(Speech2Text, args, "phonemize")
    lang_sym = f"<{args.language}>" if args.language else _no_language(s2t)

    if s2t.ctc_only:
        decoded = _decode(s2t, args.audio, lang_sym, "<pr>")
    else:
        # An encoder-decoder checkpoint segments long audio by its own
        # timestamps, which is right for a transcript and wrong here. A
        # window shorter than the model's is padded with silence, and asked
        # for phones over that silence POWSM repeats what it has already
        # said, for as long as the window lasts. One window at a time
        # instead, each decoded on its own, which is what the model card
        # does.
        decoded = " ".join(
            s2t(window, lang_sym=lang_sym, task_sym="<pr>")[0][0]
            for window in _windows(s2t, args.audio)
        )
    print(phones(decoded, spaced=args.spaced))
    return 0


def _windows(s2t, audio: str):
    """The recording in pieces of the length the checkpoint was trained on."""
    speech = s2t.read_audio(audio)
    length = int(s2t.preprocessor_conf["speech_length"] * s2t.preprocessor_conf["fs"])
    for start in range(0, max(len(speech), 1), length):
        yield speech[start : start + length]


def cmd_align(args) -> int:
    _require_file(args.audio)
    if not args.text and not args.text_file:
        raise CLIError("give --text once per utterance, or --text-file")
    if args.text and args.text_file:
        raise CLIError("give --text or --text-file, not both")
    if args.text_file:
        _require_file(args.text_file)
        utterances = [
            line.strip()
            for line in Path(args.text_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        utterances = list(args.text)
    if not utterances:
        raise CLIError(f"{args.text_file} has no lines to align")

    from espnet2.bin.align import ForcedAligner

    aligner = _build(ForcedAligner, args, "align")
    try:
        segments = aligner(args.audio, utterances)
    except ValueError as e:
        # "this text cannot fit in this recording", and the like: things the
        # user can correct, rather than a traceback
        raise CLIError(str(e)) from e

    # start, end, how sure it is, and the words: a line a person can read and
    # a line `cut` can take apart
    def line(piece, indent=""):
        return (
            f"{indent}{piece.start:.2f}\t{piece.end:.2f}\t"
            f"{piece.score:.4f}\t{piece.text}"
        )

    for segment in segments:
        print(line(segment))
        if args.tokens:
            for token in segment.tokens:
                print(line(token, indent="  "))
    return 0


def cmd_translate(args) -> int:
    _require_file(args.audio)
    from espnet2.bin.s2t_inference import Speech2Text

    s2t = _build(Speech2Text, args, "translate")
    print(_decode(s2t, args.audio, f"<{args.language}>", f"<st_{args.to}>"))
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


def cmd_synthesize(args) -> int:
    path = _output_path(args.output)
    from espnet2.bin.tts_inference import Text2Speech

    tts = _build(Text2Speech, args, "synthesize")
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
    from espnet2.bin.s2t_inference import Speech2Text

    s2t = _build(Speech2Text, args, "demo")
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

    def add(name, help_text, func, device="cpu", was=None):
        # `was` is the name this command had in a release. argparse aliases
        # share one parser, so the old name keeps working exactly, and the
        # help listing says "transcribe (asr)" - which is where someone
        # looking for the name they remember will look.
        p = sub.add_parser(name, aliases=[was] if was else [], help=help_text)
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

    p = add("transcribe", "transcribe an audio file", cmd_transcribe, was="asr")
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

    p = add("phonemize", "recognise the phones in an audio file", cmd_phonemize)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument(
        "--language",
        default=None,
        # Worth naming: on test_utils/ctc_align_test.wav, POWSM-CTC answers
        # `dəseɪlʌvðəhotɛlsɪz` without it and `ðəseɪlʌvðəhoʊtɛlzɪz` with
        # --language eng - English r and diphthongs rather than a tap and
        # plain vowels.
        help="POWSM language token, ISO 639-3: eng, jpn, deu … Name it if you "
        "know it; without it the model is told the language is unknown, which "
        "it handles but reads less like the language",
    )
    p.add_argument(
        "--spaced",
        action="store_true",
        help="one phone at a time, separated by spaces, rather than as IPA",
    )

    p = add("align", "line text up with the audio it was said in", cmd_align)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument(
        "--text",
        action="append",
        default=[],
        help="one utterance; give it once per utterance, in the order spoken",
    )
    p.add_argument(
        "--text-file", help="a file with one utterance a line, instead of --text"
    )
    p.add_argument(
        "--tokens",
        action="store_true",
        help="also print each token's own start, end and probability",
    )

    p = add("translate", "translate speech into another language", cmd_translate)
    p.add_argument("audio", help="audio file, any format soundfile reads")
    p.add_argument("--to", required=True, help="OWSM target language token, ISO 639-3")
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="OWSM language token of the speech, ISO 639-3 (default: detected)",
    )

    p = add("synthesize", "synthesise speech from text", cmd_synthesize, was="tts")
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


# What each command was called in a release. Both names work; this is only
# so that the old one says where it went, once, on stderr.
RENAMED = {"asr": "transcribe", "tts": "synthesize"}


def main(argv: Optional[List[str]] = None) -> int:
    typed = next(
        (
            a
            for a in (argv if argv is not None else sys.argv[1:])
            if not a.startswith("-")
        ),
        None,
    )
    if typed in RENAMED:
        print(
            f"espnet: `{typed}` is now `{RENAMED[typed]}`, for one name a task "
            f"rather than two. The old name still works.",
            file=sys.stderr,
        )
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
