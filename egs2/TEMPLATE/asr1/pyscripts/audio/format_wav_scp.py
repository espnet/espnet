#!/usr/bin/env python3
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import humanfriendly
import kaldiio
import numpy as np
import resampy
import soundfile
from tqdm import tqdm
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.sound_scp import SoundScpWriter, soundfile_read
from espnet2.fileio.vad_scp import VADScpReader
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """

    >>> str2int_tuple('3,4,5')
    (3, 4, 5)

    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def vad_trim(vad_reader: VADScpReader, uttid: str, wav: np.array, fs: int) -> np.array:
    # Conduct trim wtih vad information

    assert check_argument_types()
    assert uttid in vad_reader, uttid

    vad_info = vad_reader[uttid]
    total_length = sum(int((time[1] - time[0]) * fs) for time in vad_info)
    new_wav = np.zeros((total_length,), dtype=wav.dtype)
    start_frame = 0
    for time in vad_info:
        # Note: we regard vad as [xxx, yyy)
        duration = int((time[1] - time[0]) * fs)
        orig_start_frame = int(time[0] * fs)
        orig_end_frame = orig_start_frame + duration

        end_frame = start_frame + duration
        new_wav[start_frame:end_frame] = wav[orig_start_frame:orig_end_frame]

        start_frame = end_frame

    return new_wav


class SegmentsExtractor:
    """Emulating kaldi extract-segments.cc

    Args:
        segments (str): The file format is
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
    """

    def __init__(self, fname: str, segments: str = None, multi_columns: bool = False):
        assert check_argument_types()
        self.wav_scp = fname
        self.multi_columns = multi_columns
        self.wav_dict = {}
        with open(self.wav_scp, "r") as f:
            for line in f:
                recodeid, wavpath = line.strip().split(None, 1)
                if recodeid in self.wav_dict:
                    raise RuntimeError(f"{recodeid} is duplicated")
                self.wav_dict[recodeid] = wavpath

        self.segments = segments
        self.segments_dict = {}
        with open(self.segments, "r") as f:
            for line in f:
                sps = line.rstrip().split(None)
                if len(sps) != 4:
                    raise RuntimeError("Format is invalid: {}".format(line))
                uttid, recodeid, st, et = sps
                self.segments_dict[uttid] = (recodeid, float(st), float(et))

                if recodeid not in self.wav_dict:
                    raise RuntimeError(
                        'Not found "{}" in {}'.format(recodeid, self.wav_scp)
                    )

    def generator(self):
        recodeid_counter = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            recodeid_counter[recodeid] = recodeid_counter.get(recodeid, 0) + 1

        cached = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            wavpath = self.wav_dict[recodeid]
            if recodeid not in cached:
                if wavpath.endswith("|"):
                    if self.multi_columns:
                        raise RuntimeError(
                            "Not supporting multi_columns wav.scp for inputs by pipe"
                        )
                    # Streaming input e.g. cat a.wav |
                    with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                        with BytesIO(f.read()) as g:
                            array, rate = soundfile.read(g)

                else:
                    if self.multi_columns:
                        array, rate = soundfile_read(
                            wavs=wavpath.split(),
                            dtype=None,
                            always_2d=False,
                            concat_axis=1,
                        )
                    else:
                        array, rate = soundfile.read(wavpath)
                cached[recodeid] = array, rate

            array, rate = cached[recodeid]
            # Keep array until the last query
            recodeid_counter[recodeid] -= 1
            if recodeid_counter[recodeid] == 0:
                cached.pop(recodeid)
            # Convert starting time of the segment to corresponding sample number.
            # If end time is -1 then use the whole file starting from start time.
            if et != -1:
                array = array[int(st * rate) : int(et * rate)]
            else:
                array = array[int(st * rate) :]

            yield utt, (array, rate), None, None


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Create waves list from "wav.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="wav",
        help='Specify the prefix word of output file name such as "wav.scp"',
    )
    parser.add_argument("--segments", default=None)
    parser.add_argument(
        "--fs",
        type=humanfriendly_or_none,
        default=None,
        help="If the sampling rate specified, Change the sampling rate.",
    )
    parser.add_argument("--audio-format", default="wav")
    parser.add_argument("--vad_based_trim", type=str, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ref-channels", default=None, type=str2int_tuple)
    group.add_argument("--utt2ref-channels", default=None, type=str)
    group.add_argument(
        "--audio-subtype",
        default=None,
        type=str,
        help=(
            "Give a interpretable subtype by soundfile e.g. PCM_16. "
            "You can check all available types by soundfile.available_subtypes()"
        ),
    )
    parser.add_argument(
        "--multi-columns-input",
        type=str2bool,
        default=False,
        help=(
            "Enable multi columns mode for input wav.scp. "
            "e.g. 'ID a.wav b.wav c.wav' is interpreted as 3ch audio data"
        ),
    )
    parser.add_argument(
        "--multi-columns-output",
        type=str2bool,
        default=False,
        help=(
            "Enable multi columns mode for output wav.scp. "
            "e.g. If input audio data has 2ch, "
            "each line in wav.scp has the the format like "
            "'ID ID-CH0.wav ID-CH1.wav'"
        ),
    )
    args = parser.parse_args()

    out_num_samples = Path(args.outdir) / "utt2num_samples"

    if args.ref_channels is not None:

        def utt2ref_channels(x) -> Tuple[int, ...]:
            return args.ref_channels

    elif args.utt2ref_channels is not None:
        utt2ref_channels_dict = read_2columns_text(args.utt2ref_channels)

        def utt2ref_channels(x, d=utt2ref_channels_dict) -> Tuple[int, ...]:
            chs_str = d[x]
            return tuple(map(int, chs_str.split()))

    else:
        utt2ref_channels = None

    if args.audio_format.endswith("ark") and args.multi_columns_output:
        raise RuntimeError("Multi columns wav.scp is not supported for ark type")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_wavscp = Path(args.outdir) / f"{args.name}.scp"

    if args.audio_format.endswith("ark"):
        fark = open(Path(args.outdir) / f"data_{args.name}.ark", "wb")
        fscp_out = out_wavscp.open("w")
        writer = None
    else:
        writer = SoundScpWriter(
            args.outdir,
            out_wavscp,
            format=args.audio_format,
            multi_columns=args.multi_columns_output,
            subtype=args.audio_subtype,
        )
        fscp_out = None

    if args.vad_based_trim is not None:
        vad_reader = VADScpReader(args.vad_based_trim)

    if args.segments is not None:
        extractor = SegmentsExtractor(
            args.scp, segments=args.segments, multi_columns=args.multi_columns_input
        )
        generator = extractor.generator

    else:

        def generator():
            with Path(args.scp).open("r") as fscp:
                for line in tqdm(fscp):
                    uttid, wavpath = line.strip().split(None, 1)

                    # B.a. Without segments and using pipe inputs
                    if wavpath.endswith("|"):
                        if args.multi_columns_input:
                            raise RuntimeError(
                                "Not supporting multi_columns wav.scp for inputs by"
                                " pipe"
                            )
                        # Streaming input e.g. cat a.wav |
                        with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                            with BytesIO(f.read()) as g:
                                wave, rate = soundfile.read(g)
                        subtypes = None

                    # B.b Without segments and not using pipe
                    else:
                        if args.multi_columns_input:
                            wave, rate, subtypes = soundfile_read(
                                wavs=wavpath.split(),
                                dtype=None,
                                always_2d=False,
                                concat_axis=1,
                                return_subtype=True,
                            )
                        else:
                            with soundfile.SoundFile(wavpath) as sf:
                                rate = sf.samplerate
                                subtypes = [sf.subtype]
                                wave = sf.read()
                    yield uttid, (wave, rate), wavpath, subtypes

    with out_num_samples.open("w") as fnum_samples:
        for uttid, (wave, rate), wavpath, subtypes in tqdm(generator()):
            save_asis = True
            if args.fs is not None and args.fs != rate:
                # FIXME(kamo): To use sox?
                wave = resampy.resample(wave, rate, args.fs, axis=0)
                rate = args.fs
                save_asis = False

            if args.vad_based_trim is not None:
                wave = vad_trim(vad_reader, uttid, wave, rate)
                save_asis = False

            if wave.ndim == 2 and utt2ref_channels is not None:
                wave = wave[:, utt2ref_channels(uttid)]
                save_asis = False

            if args.segments is not None:
                save_asis = False

            if args.audio_format.endswith("ark"):
                save_asis = False

            if args.multi_columns_input:
                if args.multi_columns_output:
                    if wavpath is not None:
                        for _wavpath in wavpath.split():
                            if Path(_wavpath).suffix != "." + args.audio_format:
                                save_asis = False
                                break

                        if wave.ndim == 1:
                            _num_ch = 1
                        else:
                            _num_ch = wave.shape[1]
                        if len(wavpath.split()) != _num_ch:
                            save_asis = False
                else:
                    if wavpath is not None and len(wavpath.split()) > 1:
                        save_asis = False

            elif args.multi_columns_output:
                if wave.ndim == 2 and wave.shape[1] > 1:
                    save_asis = False

            if wavpath is not None and wavpath.endswith("|"):
                save_asis = False
            if wavpath is not None and Path(wavpath).suffix != "." + args.audio_format:
                save_asis = False

            if not args.audio_format.endswith("ark") and subtypes is not None:
                if args.audio_subtype is None:
                    subtype2 = soundfile.default_subtype(args.audio_format)
                else:
                    subtype2 = args.audio_subtype
                for subtype in subtypes:
                    if subtype != subtype2:
                        save_asis = False
                        break

            if save_asis:
                writer.fscp.write(f"{uttid} {wavpath}\n")

            elif args.audio_format.endswith("ark"):
                for name in soundfile.available_formats():
                    if name.lower() in args.audio_format.lower():
                        suf = name.lower()
                        break
                else:
                    raise RuntimeError(f"{args.audio_format} is not supported.")

                # NOTE(kamo): Using extended ark format style here.
                # This format is incompatible with Kaldi
                kaldiio.save_ark(
                    fark,
                    {uttid: (wave, rate)},
                    scp=fscp_out,
                    append=True,
                    write_function="soundfile",
                    write_kwargs={"format": suf, "subtype": args.audio_subtype},
                )

            else:
                writer[uttid] = rate, wave
            fnum_samples.write(f"{uttid} {len(wave)}\n")


if __name__ == "__main__":
    main()
