"""
Slighly modified versions of lhotse recipes scripts in
https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/
"""

import glob
import os.path
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union
import argparse
import json
import soundfile as sf

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations, safe_extract, urlretrieve_progress
from generate_data import choose_txt_normalization


def prepare_chime6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: str ="dev",
    mic: str = "mdm",
    normalize_text: str = "chime6",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of CHiME-6 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use, choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings. For MDM, there are 6 array devices with 4
        channels each, so the resulting recordings will have 24 channels.
    :param normalize_text: str, the text normalization method, choose from "chime6" or "chime7".
    """
    import soundfile as sf
    assert mic in ["ihm", "mdm"], "mic must be either 'ihm' or 'mdm'."
    txt_normalization = choose_txt_normalization(normalize_text)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = [Path(x).stem for x in glob.glob(os.path.join(corpus_dir, "transcriptions", dset_part, "*.json"))]
    manifests = defaultdict(dict)
    recordings = []
    supervisions = []

    # First we create the recordings
    if mic == "ihm":
        global_spk_channel_map = {}
        for session in all_sessions:
            audio_paths = [Path(x) for x in glob.glob(os.path.join(corpus_dir, "audio", dset_part, f"{session}_P*.wav"))]
            if len(audio_paths) == 0:
                raise FileNotFoundError(
                    f"No audio found for session {session} in {dset_part} set.")
            sources = []
            # NOTE: Each headset microphone is binaural
            for idx, audio_path in enumerate(audio_paths):
                sources.append(
                    AudioSource(
                        type="file",
                        channels=[2 * idx, 2 * idx + 1],
                        source=str(audio_path)))
                spk_id = audio_path.stem.split("_")[1]
                global_spk_channel_map[(session, spk_id)] = [2 * idx, 2 * idx + 1]
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
    else:
        for session in all_sessions:
            audio_paths = [Path(x) for x in glob.glob(os.path.join(corpus_dir, "audio", dset_part, f"{session}_U*.wav"))]
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

    def _get_channel(spk_id, session, ref=None):
        if mic == "ihm":
            return global_spk_channel_map[(session, spk_id)]
        else:
            recording = recordings[session]
            return (
                list(range(recording.num_channels))
                if not ref
                else [i for i, s in enumerate(recording.sources) if ref in s.source]
            )

    # Then we create the supervisions
    for session in all_sessions:
        with open(os.path.join(corpus_dir, "transcriptions", dset_part, f"{session}.json")) as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                channel = _get_channel(
                    spk_id,
                    session,
                    ref=None,
                )
                start = float(segment["start_time"])
                end = float(segment["end_time"])
                if start >= end:  # some segments may have negative duration
                    continue
                supervisions.append(
                    SupervisionSegment(
                        id=f"{session}-{idx}",
                        recording_id=session,
                        start=start,
                        duration=add_durations(end, -start, sampling_rate=16000),
                        channel=channel,
                        text=txt_normalization(
                            segment["words"]),
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
        output_dir / f"CHiME6-{mic}_supervisions_{dset_part}.jsonl.gz")
    recording_set.to_file(
        output_dir / f"CHiME6-{mic}_recordings_{dset_part}.jsonl.gz")

    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set}
    return manifests


def prepare_dipco(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part="dev",
    mic: Optional[str] = "mdm",
    normalize_text: Optional[str] = "chime6",
    json_dir: Optional[Pathlike] = None, # alternative annotation e.g. from non-oracle diarization  #TODO
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of DiPCo main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use, choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings. For MDM, there are 5 array devices with 7
        channels each, so the resulting recordings will have 35 channels.
    :param normalize_text: str, the text normalization method, choose from "chime6" or "chime7".
    """

    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"
    normalize_text_func = choose_txt_normalization(normalize_text)

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    all_sessions = glob.glob(os.path.join(corpus_dir,
                                          "transcriptions",
                                          dset_part,
                                          "*.json"))
    all_sessions = [Path(x).stem for x in all_sessions]
    recordings = []
    supervisions = []

    # First we create the recordings
    if mic == "ihm":
        global_spk_channel_map = {}
        for session in all_sessions:
            audio_paths = [Path(x) for x in glob.glob(os.path.join(corpus_dir, "audio", dset_part,
                                                                   f"{session}_P*.wav"))]
            sources = []
            for idx, audio_path in enumerate(audio_paths):
                sources.append(
                    AudioSource(type="file", channels=[idx], source=str(audio_path))
                )
                spk_id = audio_path.stem.split("_")[1]
                global_spk_channel_map[(session, spk_id)] = idx

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
    else:
        for session in all_sessions:
            audio_paths = [
                p for p in (corpus_dir / "audio" / dset_part).rglob(f"{session}_U*.wav")
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
        with open(corpus_dir / "transcriptions" / dset_part / f"{session}.json") as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                channel = (
                    global_spk_channel_map[(session, spk_id)]
                    if mic == "ihm"
                    else list(range(35))
                )
                start = float(segment["start_time"])
                end = float(segment["end_time"])
                supervisions.append(
                    SupervisionSegment(
                        id=f"{session}-{idx}",
                        recording_id=session,
                        start=start,
                        duration=add_durations(end, -start, sampling_rate=16000),
                        channel=channel,
                        text=normalize_text_func(
                            segment["words"]),
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
            output_dir / f"DiPCo-{mic}_supervisions_{dset_part}.jsonl.gz"
        )
    recording_set.to_file(
            output_dir / f"DiPCo-{mic}_recordings_{dset_part}.jsonl.gz"
        )

    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Lhotse manifests generation scripts for CHiME-7 Task 1 data.",
                                     add_help=True, usage='%(prog)s [options]')
    parser.add_argument("-c,--chime7task1_root", type=str, metavar='STR', dest="chime7_root",
                        help="Path to CHiME-7 Task 1 dataset main directory, as generated by the create_dataset.sh script."
                             "It should contain chime6, dipco and mixer6 as sub-folders.")
    parser.add_argument("-d,--dset_part", type=str, metavar='STR', dest="dset_part",
                        help="Which part of the dataset you want for the manifests? 'train', 'dev' or 'eval' ?")
    parser.add_argument("-o,--output_root", type=str, metavar='STR', dest="output_root",
                        help="Path where the new CHiME-7 Task 1 dataset will be saved."
                             "Note that for audio files symbolic links are used.")
    parser.add_argument("--txt_norm", type=str, default="chime6", metavar='STR',
                        help="Choose between chime6 and chime7, this select the text normalization applied when creating" \
                             "the scoring annotation.")
    parser.add_argument("-c,--diar_jsons_root", type=str, metavar='STR', dest="diar_json",
                        help="Path to a directory with same structure as CHiME-7 Task 1 transcription directory, containing JSON files"
                             "with the same structure but from a diarization system (words field could be missing and will be ignored).")
    args = parser.parse_args()
    assert args.dset_part in ["train", "dev", "eval"], "Option --dset_part should be 'train', 'dev' or 'eval'"
    if args.dset_part in ["train", "dev"]:
        valid_mics = ["mdm", "ihm"]
    elif args.dset_part == "eval":
        valid_mics = ["mdm"]
        for mic in valid_mics:
            prepare_chime6(os.path.join(args.chime7_root, "chime6"),
                           os.path.join(args.output_root, "chime6",
                                        args.dset_part), args.dset_part, mic=mic)
            if args.dset_part != "train":
                # dipco has no training set.
                prepare_dipco(os.path.join(args.chime7_root, "dipco"),
                          os.path.join(args.output_root, "dipco",
                                        args.dset_part), args.dset_part, mic=mic)
            prepare_mixer6(os.path.join(args.chime7_root, "mixer6"),
                          os.path.join(args.output_root, "mixer6",
                                        args.dset_part), args.dset_part, mic=mic)






