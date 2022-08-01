#!/usr/bin/env python3
import argparse
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile


def prepare(
    dirha_dir: str,
    audio_dir: str,
    data_dir: Optional[str],
    audio_format: str = "flac",
):
    dirha_dir = Path(dirha_dir)
    audio_dir = Path(audio_dir)
    if data_dir is not None:
        data_dir = Path(data_dir)

    for env, mics in [
        ("Kitchen/Circular_Array", ["KA1", "KA2", "KA3", "KA4", "KA5", "KA6"]),
        (
            "Livingroom/Circular_Array",
            ["Beam_Circular_Array", "LA1", "LA2", "LA3", "LA4", "LA5", "LA6"],
        ),
        (
            "Livingroom/Linear_Array",
            [
                "Beam_Linear_Array",
                "LD02",
                "LD03",
                "LD04",
                "LD05",
                "LD06",
                "LD07",
                "LD08",
                "LD09",
                "LD10",
                "LD11",
                "LD12",
            ],
        ),
        (
            "Livingroom/Wall",
            ["L1C", "L1L", "L1R", "L2L", "L2R", "L3L", "L3R", "L4L", "L4R"],
        ),
    ]:
        for real_sim in ["Real", "Sim"]:
            info = {}
            d = dirha_dir / "data" / "DIRHA_English_wsj" / real_sim
            if not d.exists():
                raise FileNotFoundError(f"{d} is not found")

            for p in d.glob("*"):
                p = p / env
                for mic in mics:
                    # Create Kaldi style datadir, also
                    _info, _spk2utt, _spk2gender = info.setdefault(
                        (env, real_sim, mic), ({}, {}, {})
                    )

                    xml = f"{p / mic}.xml"
                    wav = f"{p / mic}.wav"
                    for k in ET.parse(xml).findall("SOURCE"):
                        # Read XML file
                        name = k.find("name").text
                        spk_id = k.find("spk_id").text
                        uid = f"{spk_id}-{name}"
                        gender = k.find("gender").text
                        begin = k.find("begin_sample").text
                        end = k.find("end_sample").text
                        txt = k.find("txt").text.upper()

                        # Read, chunk, and write audio file
                        adir = (
                            audio_dir / f"DIRHA_wsj_oracle_VAD_{real_sim.lower()}_"
                            f"{env.replace('/', '_')}_{mic}" / spk_id / gender / name
                        )
                        adir.parent.mkdir(parents=True, exist_ok=True)
                        w, r = soundfile.read(wav, start=int(begin) - 1, stop=int(end))
                        out_audio = f"{adir}.{audio_format}"
                        soundfile.write(out_audio, w, r)

                        _info[uid] = [str(out_audio), spk_id, txt]
                        _spk2utt.setdefault(spk_id, []).append(uid)
                        _spk2gender[spk_id] = gender

            if data_dir is not None:
                # Create single channel dir
                env2info = {}
                for (env, real_sim, mic), (
                    _info,
                    _spk2utt,
                    _spk2gender,
                ) in info.items():
                    if not mic.startswith("Beam_"):
                        env2info.setdefault((env, real_sim), []).append(
                            (_info, _spk2utt, _spk2gender)
                        )

                    ddir = (
                        data_dir
                        / f"dirha_{real_sim.lower()}_{env.replace('/', '_')}_{mic}"
                    )
                    ddir.mkdir(parents=True, exist_ok=True)
                    fs = [(ddir / t).open("w") for t in ["wav.scp", "utt2spk", "text"]]
                    for name in sorted(list(_info)):
                        for v, f in zip(_info[name], fs):
                            f.write(f"{name} {v}\n")
                    for f in fs:
                        f.close()
                    with (ddir / "spk2gender").open("w") as f:
                        for spk in sorted(list(_spk2gender)):
                            gender = _spk2gender[spk]
                            f.write(f"{spk} {gender[0]}\n")

                    with (ddir / "spk2utt").open("w") as f:
                        for spk in sorted(list(_spk2utt)):
                            utts = sorted(_spk2utt[spk])
                            f.write(f"{spk} {' '.join(utts)}\n")

                # Create multi channel dir
                for (env, real_sim), info_list in env2info.items():
                    ddir = (
                        data_dir
                        / f"dirha_{real_sim.lower()}_{env.replace('/', '_')}_multi"
                    )
                    ddir.mkdir(parents=True, exist_ok=True)

                    _info, _spk2utt, _spk2gender = info_list[0]

                    with (ddir / "wav.scp").open("w") as wavf:
                        for name in _info:
                            # Get output file names for each mic
                            files = [i[name][0] for i, _, _ in info_list]
                            wavs = [soundfile.read(f) for f in files]

                            # Create multi channel audio
                            if not all(len(wavs[0][0]) == len(wav[0]) for wav in wavs):
                                warnings.warn(
                                    f"{env}, {real_sim}, name={name} doesn't"
                                    " have same length audio for each channel. "
                                    "Truncated to the shortest one."
                                )
                                s = min([len(wav[0]) for wav in wavs])
                                wavs = [(wav[0][:s], wav[1]) for wav in wavs]

                            wav = np.stack([wav[0] for wav in wavs], axis=1)
                            fs = wavs[0][1]
                            spk_id = _info[name][1]
                            gender = _spk2gender[spk_id]

                            adir = (
                                audio_dir / f"DIRHA_wsj_oracle_VAD_{real_sim.lower()}_"
                                f"{env.replace('/', '_')}_multi"
                                / spk_id
                                / gender
                                / name
                            )
                            adir.parent.mkdir(parents=True, exist_ok=True)
                            if audio_format == "flac":
                                # flac doesn't support 9 or more channels
                                if wav.shape[1] > 8:
                                    _audio_format = "wav"
                                else:
                                    _audio_format = "flac"
                            else:
                                _audio_format = audio_format
                            out_audio = f"{adir}.{_audio_format}"
                            soundfile.write(out_audio, wav, fs)
                            wavf.write(f"{name} {out_audio}\n")

                    with (ddir / "utt2spk").open("w") as f:
                        for name in _info:
                            v = _info[name][1]
                            f.write(f"{name} {v}\n")
                    with (ddir / "text").open("w") as f:
                        for name in _info:
                            v = _info[name][2]
                            f.write(f"{name} {v}\n")
                    with (ddir / "spk2gender").open("w") as f:
                        for spk in sorted(list(_spk2gender)):
                            gender = _spk2gender[spk]
                            f.write(f"{spk} {gender[0]}\n")
                    with (ddir / "spk2utt").open("w") as f:
                        for spk in sorted(list(_spk2utt)):
                            utts = sorted(_spk2utt[spk])
                            f.write(f"{spk} {' '.join(utts)}\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Dirha WSJ data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dirha_dir", required=True, help="Input directory")
    parser.add_argument("--audio_dir", required=True, help="Output directory")
    parser.add_argument("--data_dir", help="Output directory")
    parser.add_argument("--audio_format", default="flac")
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    prepare(**kwargs)


if __name__ == "__main__":
    main()