"""
Slighly modified versions of lhotse recipes scripts in
https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/
"""
import argparse
import glob
import json
import os.path
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
import soundfile as sf
from gen_task1_data import choose_txt_normalization
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations, safe_extract, urlretrieve_progress


def prepare_chime6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: str = "dev",
    mic: str = "mdm",
    normalize_text: str = "chime6",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    ignore_shorter: Optional[float] = 0.2,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of CHiME-6 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 6 array devices with 4
        channels each, so the resulting recordings will have 24 channels.
    :param normalize_text: str, the text normalization method,
    choose from "chime6" or "chime7".
    """
    import soundfile as sf

    assert mic in ["ihm", "mdm"], "mic must be either 'ihm' or 'mdm'."
    txt_normalization = choose_txt_normalization(normalize_text)
    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = [
        Path(x).stem
        for x in glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    ]
    manifests = defaultdict(dict)
    recordings = []
    supervisions = []
    # First we create the recordings
    if mic == "ihm":
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_P*.wav")
                )
            ]
            if len(audio_paths) == 0:
                raise FileNotFoundError(
                    f"No audio found for session {session} in {dset_part} set."
                )
            sources = []
            # NOTE: Each headset microphone is binaural
            for idx, audio_path in enumerate(audio_paths):
                channels = [0, 1]  # if dset_part == "train" else [0]
                sources.append(
                    AudioSource(type="file", channels=channels, source=str(audio_path))
                )
                spk_id = audio_path.stem.split("_")[1]
                audio_sf = sf.SoundFile(str(audio_paths[0]))
                recordings.append(
                    Recording(
                        id=session + f"_{spk_id}",
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )
    else:
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_U*.wav")
                )
            ]
            # discard some arrays because their
            # files length is a lot different and causes GSS to fail
            if session == "S12":
                audio_paths = [
                    x for x in audio_paths if not Path(x).stem.startswith("S12_U05")
                ]
            elif session == "S24":
                audio_paths = [
                    x for x in audio_paths if not Path(x).stem.startswith("S24_U06")
                ]
            elif session == "S18":
                audio_paths = [
                    x for x in audio_paths if not Path(x).stem.startswith("S18_U06")
                ]
            sources = []
            for idx, audio_path in enumerate(sorted(audio_paths)):
                sources.append(
                    AudioSource(type="file", channels=[idx], source=str(audio_path))
                )

            audio_sf = sf.SoundFile(str(audio_paths[0]))
            recordings.append(
                Recording(
                    id=session,
                    sources=sources,
                    sampling_rate=int(audio_sf.samplerate),
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
            )
    recordings = RecordingSet.from_recordings(recordings)

    def _get_channel(session, dset_part):
        if mic == "ihm":
            return [0, 1] if dset_part == "train" else [0]
        else:
            recording = recordings[session]
            return list(range(recording.num_channels))

    # Then we create the supervisions
    for session in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{session}.json")) as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                channel = _get_channel(session, dset_part)
                start = float(segment["start_time"])
                end = float(segment["end_time"])
                if ignore_shorter is not None and (end - start) < ignore_shorter:
                    print(
                        "Ignored segment session {} speaker "
                        "{} seconds {} to {}, because shorter than {}"
                        "".format(session, spk_id, start, end, ignore_shorter)
                    )
                    continue
                if start >= end:  # some segments may have negative duration
                    continue

                ex_id = (
                    f"{spk_id}_chime6_{session}_{idx}-"
                    f"{round(100*start):06d}_{round(100*end):06d}-{mic}"
                )

                if "words" not in segment.keys():
                    assert json_dir is not None
                    segment["words"] = "placeholder"

                supervisions.append(
                    SupervisionSegment(
                        id=ex_id,
                        recording_id=session
                        if mic == "mdm"
                        else session + f"_{spk_id}",
                        start=start,
                        duration=add_durations(end, -start, sampling_rate=16000),
                        channel=channel,
                        text=txt_normalization(segment["words"]),
                        language="English",
                        speaker=spk_id,
                    )
                )

    supervisions = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(
        recordings=recordings, supervisions=supervisions
    )
    # Fix manifests
    validate_recordings_and_supervisions(recording_set, supervision_set)
    supervision_set.to_file(
        os.path.join(output_dir, f"chime6-{mic}_supervisions_{dset_part}.jsonl.gz")
    )
    recording_set.to_file(
        os.path.join(output_dir, f"chime6-{mic}_recordings_{dset_part}.jsonl.gz")
    )
    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests


def prepare_dipco(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: Optional[str] = "dev",
    mic: Optional[str] = "mdm",
    normalize_text: Optional[str] = "chime6",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    ignore_shorter: Optional[float] = 0.2,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of DiPCo main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 5 array devices with 7
        channels each, so the resulting recordings will have 35 channels.
    :param normalize_text: str, the text normalization method,
     choose from "chime6" or "chime7".
    """
    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"
    normalize_text_func = choose_txt_normalization(normalize_text)
    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    all_sessions = glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    all_sessions = [Path(x).stem for x in all_sessions]
    recordings = []
    supervisions = []
    # First we create the recordings
    if mic == "ihm":
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_P*.wav")
                )
            ]
            # sources = []
            for idx, audio_path in enumerate(audio_paths):
                sources = [
                    AudioSource(type="file", channels=[0], source=str(audio_path))
                ]
                spk_id = audio_path.stem.split("_")[1]
                audio_sf = sf.SoundFile(str(audio_path))
                recordings.append(
                    Recording(
                        id=session + "_{}".format(spk_id),
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )
    else:
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_U*.wav")
                )
            ]
            sources = []
            for idx, audio_path in enumerate(sorted(audio_paths)):
                sources.append(
                    AudioSource(type="file", channels=[idx], source=str(audio_path))
                )

            audio_sf = sf.SoundFile(str(audio_paths[0]))
            recordings.append(
                Recording(
                    id=session,
                    sources=sources,
                    sampling_rate=int(audio_sf.samplerate),
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
            )

    # Then we create the supervisions
    for session in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{session}.json")) as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                channel = [0] if mic == "ihm" else list(range(35))
                start = float(segment["start_time"])
                end = float(segment["end_time"])
                if ignore_shorter is not None and (end - start) < ignore_shorter:
                    print(
                        "Ignored segment session {} speaker {} "
                        "seconds {} to {}, because shorter than {}"
                        "".format(session, spk_id, start, end, ignore_shorter)
                    )
                    continue

                ex_id = (
                    f"{spk_id}_dipco_{session}_{idx}-"
                    f"{round(100 * start):06d}_{round(100 * end):06d}-{mic}"
                )
                if "words" not in segment.keys():
                    assert json_dir is not None
                    segment["words"] = "placeholder"
                supervisions.append(
                    SupervisionSegment(
                        id=ex_id,
                        recording_id=session
                        if mic == "mdm"
                        else session + "_{}".format(spk_id),
                        start=start,
                        duration=add_durations(end, -start, sampling_rate=16000),
                        channel=channel,
                        text=normalize_text_func(segment["words"]),
                        speaker=spk_id,
                    )
                )

    recording_set, supervision_set = fix_manifests(
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )
    # Fix manifests
    validate_recordings_and_supervisions(recording_set, supervision_set)
    supervision_set.to_file(
        os.path.join(output_dir, f"dipco-{mic}_supervisions_{dset_part}.jsonl.gz")
    )
    recording_set.to_file(
        os.path.join(output_dir, f"dipco-{mic}_recordings_{dset_part}.jsonl.gz")
    )
    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests


def prepare_mixer6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part="dev",
    mic: Optional[str] = "mdm",
    normalize_text: Optional[str] = "chime6",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    ignore_shorter: Optional[float] = 0.2,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"
    if mic == "ihm":
        assert dset_part in [
            "train_intv",
            "train_call",
            "dev",
        ], "No close-talk microphones on evaluation set."

    normalize_text_func = choose_txt_normalization(normalize_text)
    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    all_sessions = [Path(x).stem for x in all_sessions]
    audio_files = glob.glob(os.path.join(corpus_dir, "audio", dset_part, "*.flac"))
    assert len(audio_files) > 0, "Can't parse mixer6 audio files, is the path correct ?"
    sess2audio = {}
    for audio_f in audio_files:
        sess_name = "_".join(Path(audio_f).stem.split("_")[:-1])
        if sess_name not in sess2audio.keys():
            sess2audio[sess_name] = [audio_f]
        else:
            sess2audio[sess_name].append(audio_f)

    recordings = []
    supervisions = []
    for sess in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{sess}.json")) as f:
            transcript = json.load(f)
        if mic == "ihm":
            if dset_part.startswith("train"):
                if mic == "ihm" and dset_part.startswith("train"):
                    current_sess_audio = [
                        x
                        for x in sess2audio[sess]
                        if Path(x).stem.split("_")[-1] in ["CH02"]
                    ]  # only interview and call

            elif dset_part == "dev":
                current_sess_audio = [
                    x
                    for x in sess2audio[sess]
                    if Path(x).stem.split("_")[-1] in ["CH02", "CH01"]
                ]  #
            else:
                raise NotImplementedError("No close-talk mics for eval set")

        elif mic == "mdm":
            current_sess_audio = [
                x
                for x in sess2audio[sess]
                if Path(x).stem.split("_")[-1] not in ["CH01", "CH02", "CH03"]
            ]
        else:
            raise NotImplementedError

        # recordings here
        sources = [
            AudioSource(type="file", channels=[idx], source=str(audio_path))
            for idx, audio_path in enumerate(current_sess_audio)
        ]
        audio_sf = sf.SoundFile(str(current_sess_audio[0]))
        recordings.append(
            Recording(
                id=f"{sess}-{dset_part}-{mic}",
                sources=sources,
                sampling_rate=int(audio_sf.samplerate),
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            )
        )

        for idx, segment in enumerate(transcript):
            spk_id = segment["speaker"]
            start = float(segment["start_time"])
            end = float(segment["end_time"])
            if ignore_shorter is not None and (end - start) < ignore_shorter:
                print(
                    "Ignored segment session {} speaker"
                    " {} seconds {} to {}, because shorter than {}"
                    "".format(sess, spk_id, start, end, ignore_shorter)
                )
                continue

            if mic == "ihm":  # and dset_part.startswith("train"):
                rec_id = f"{sess}-{dset_part}-{mic}"
                if mic == "ihm" and dset_part == "dev":
                    subject_id = sess.split("_")[-1]
                    if spk_id == subject_id:
                        channel = 0
                    else:
                        channel = 1
                else:
                    channel = 0
            else:
                rec_id = f"{sess}-{dset_part}-{mic}"
                channel = list(range(len(current_sess_audio)))

            ex_id = (
                f"{spk_id}_mixer6_{sess}_{dset_part}_{idx}-"
                f"{round(100 * start):06d}_{round(100 * end):06d}-{mic}"
            )
            if "words" not in segment.keys():
                assert json_dir is not None
                segment["words"] = "placeholder"
            supervisions.append(
                SupervisionSegment(
                    id=ex_id,
                    recording_id=rec_id,
                    start=start,
                    duration=add_durations(end, -start, sampling_rate=16000),
                    channel=channel,
                    text=normalize_text_func(segment["words"]),
                    speaker=spk_id,
                )
            )

    recording_set, supervision_set = fix_manifests(
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )
    # Fix manifests
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(
            os.path.join(output_dir, f"mixer6-{mic}_supervisions_{dset_part}.jsonl.gz")
        )
        recording_set.to_file(
            os.path.join(output_dir, f"mixer6-{mic}_recordings_{dset_part}.jsonl.gz")
        )

    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Lhotse manifests generation scripts for CHiME-7 Task 1 data.",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-c,--chime7task1_root",
        type=str,
        metavar="STR",
        dest="chime7_root",
        help="Path to CHiME-7 Task 1 dataset main directory,"
        " as generated by the create_dataset.sh script."
        "It should contain chime6, dipco and mixer6 as sub-folders.",
    )
    parser.add_argument(
        "-d,--d",
        type=str,
        metavar="STR",
        dest="dset_name",
        help="Name of the dataset: chime6, dipco or mixer6.",
    )
    parser.add_argument(
        "-p,--partition",
        type=str,
        metavar="STR",
        dest="dset_part",
        help="Which part of the dataset you want for "
        "the manifests? 'train', 'dev' or 'eval' ?",
    )
    parser.add_argument(
        "-o,--output_root",
        type=str,
        metavar="STR",
        dest="output_root",
        help="Path where the new CHiME-7 Task 1 dataset will be saved. "
        "Note that for audio files symbolic links are used.",
    )
    parser.add_argument(
        "--txt_norm",
        type=str,
        default="chime7",
        metavar="STR",
        required=False,
        help="Choose between chime6 and chime7, this"
        " select the text normalization applied when creating "
        "the scoring annotation.",
    )
    parser.add_argument(
        "--diar_jsons_root",
        type=str,
        metavar="STR",
        dest="diar_json",
        default="",
        required=False,
        help="Path to a directory with same structure "
        "as CHiME-7 Task 1 transcription directory, "
        "containing JSON files "
        "with the same structure but from a diarization "
        "system (words field could be missing and will be ignored).",
    )
    parser.add_argument(
        "--ignore_shorter",
        type=float,
        metavar="FLOAT",
        dest="ignore_shorter",
        default=0.0,
        required=False,
        help="Ignore segments that are shorter than this value in the supervision.",
    )

    args = parser.parse_args()

    if args.diar_json:
        diarization_json_dir = args.diar_json

        assert os.path.exists(diarization_json_dir), (
            "{} does not appear to exist"
            "did you pass the argument "
            "correctly ?".format(diarization_json_dir)
        )
    else:
        diarization_json_dir = None

    if args.diar_json:
        diarization_json_dir = args.diar_json
        assert os.path.exists(diarization_json_dir), (
            "{} does not appear to exist"
            "did you pass the argument "
            "correctly ?".format(diarization_json_dir)
        )
    else:
        diarization_json_dir = None
    assert args.dset_name in ["chime6", "dipco", "mixer6"], (
        "Datasets supported in this script " "are chime6, dipco and mixer6"
    )
    assert args.dset_part in [
        "train",
        "dev",
        "eval",
    ], "Option --dset_part should be 'train', 'dev' or 'eval'"

    if args.dset_part == "train" and args.dset_name == "dipco":
        raise NotImplementedError("DiPCo has no training set. Exiting.")

    if args.dset_part in ["train", "dev"] and diarization_json_dir is None:
        valid_mics = ["mdm", "ihm"]
    elif args.dset_part == "eval" or diarization_json_dir is not None:
        valid_mics = ["mdm"]
    for mic in valid_mics:
        if args.dset_name == "chime6":
            prepare_chime6(
                os.path.join(args.chime7_root, args.dset_name),
                os.path.join(args.output_root, args.dset_name, args.dset_part),
                args.dset_part,
                mic=mic,
                ignore_shorter=args.ignore_shorter,
                json_dir=diarization_json_dir,
            )
        elif args.dset_name == "dipco":
            prepare_dipco(
                os.path.join(args.chime7_root, args.dset_name),
                os.path.join(args.output_root, args.dset_name, args.dset_part),
                args.dset_part,
                mic=mic,
                ignore_shorter=args.ignore_shorter,
                json_dir=diarization_json_dir,
            )

        elif args.dset_name == "mixer6":
            if args.dset_part in ["eval"] and mic == "ihm":
                continue

            if args.dset_part.startswith("train"):
                supervisions = []
                recordings = []
                for dset_part in ["train_intv", "train_call"]:
                    c_manifest = prepare_mixer6(
                        os.path.join(args.chime7_root, args.dset_name),
                        None,
                        dset_part,
                        mic=mic,
                        ignore_shorter=args.ignore_shorter,
                        json_dir=diarization_json_dir,
                    )

                    supervisions.append(c_manifest[dset_part]["supervisions"])
                    recordings.append(c_manifest[dset_part]["recordings"])

                supervision_set = lhotse.combine(*supervisions)
                recording_set = lhotse.combine(*recordings)
                recording_set, supervision_set = fix_manifests(
                    RecordingSet.from_recordings(recording_set),
                    SupervisionSet.from_segments(supervision_set),
                )
                # Fix manifests
                validate_recordings_and_supervisions(recording_set, supervision_set)
                output_dir = Path(
                    os.path.join(args.output_root, args.dset_name, args.dset_part)
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                supervision_set.to_file(
                    os.path.join(
                        output_dir, f"mixer6-{mic}_supervisions_train.jsonl.gz"
                    )
                )
                recording_set.to_file(
                    os.path.join(output_dir, f"mixer6-{mic}_recordings_train.jsonl.gz")
                )
            else:
                prepare_mixer6(
                    os.path.join(args.chime7_root, args.dset_name),
                    os.path.join(args.output_root, args.dset_name, args.dset_part),
                    args.dset_part,
                    mic=mic,
                    ignore_shorter=args.ignore_shorter,
                    json_dir=diarization_json_dir,
                )
