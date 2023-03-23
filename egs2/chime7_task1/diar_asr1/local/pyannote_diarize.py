import argparse
import glob
import json
import os.path
import random
import re
from pathlib import Path

import soundfile as sf
import torch
import tqdm
from dover_lap import dover_lap
from intervaltree import IntervalTree
from pyannote.audio import Pipeline
from pyannote.metrics.segmentation import Annotation, Segment

IS_CUDA = torch.cuda.is_available()


def merge_closer(turns, distance=0.5):
    """Merge overlapping turns by same speaker within each file."""
    # Merge separately within each file and for each speaker.
    new_turns = []
    for (file_id, speaker_id), speaker_turns in dover_lap.groupby(
        turns, lambda x: (x.file_id, x.speaker_id)
    ):
        speaker_turns = list(speaker_turns)
        speaker_it = IntervalTree.from_tuples(
            [(turn.onset, turn.offset) for turn in speaker_turns]
        )
        n_turns_pre = len(speaker_it)
        speaker_it.merge_neighbors(distance=distance)
        n_turns_post = len(speaker_it)
        if n_turns_post < n_turns_pre:
            speaker_turns = []
            for intrvl in speaker_it:
                speaker_turns.append(
                    dover_lap.Turn(
                        intrvl.begin, intrvl.end, speaker_id=speaker_id, file_id=file_id
                    )
                )
            speaker_turns = sorted(speaker_turns, key=lambda x: (x.onset, x.offset))
        new_turns.extend(speaker_turns)
    return new_turns


def rttm2json(rttm_file):
    with open(rttm_file, "r") as f:
        rttm = f.readlines()

    rttm = [x.rstrip("\n") for x in rttm]
    filename = Path(rttm_file).stem

    to_json = []
    for line in rttm:
        current = line.split(" ")
        start = current[3]
        duration = current[4]
        stop = str(float(start) + float(duration))
        speaker = current[7]
        session = filename
        to_json.append(
            {
                "session_id": session,
                "speaker": speaker,
                "start_time": start,
                "end_time": stop,
            }
        )

    to_json = sorted(to_json, key=lambda x: float(x["start_time"]))
    with open(
        os.path.join(Path(rttm_file).parent, Path(rttm_file).stem + ".json"), "w"
    ) as f:
        json.dump(to_json, f, indent=4)


def ensemble_doverlap(session_id, rttms, out_dir):
    turns_list = []
    for rttm_f in rttms:
        c_turns_list = dover_lap.load_rttm(rttm_f)[0]
        for x in c_turns_list:
            x.file_id = session_id
        c_turns_list = dover_lap.merge_turns(c_turns_list)
        turns_list.append(c_turns_list)

    # Run DOVER-Lap algorithm
    file_to_out_turns = dict()
    random.shuffle(turns_list)  # We shuffle so that the hypothesis order is randomized
    file_to_out_turns[session_id] = dover_lap.DOVERLap.combine_turns_list(
        turns_list, session_id, label_mapping="hungarian"
    )

    # Write output RTTM file
    output = sum(list(file_to_out_turns.values()), [])
    output = merge_closer(output, distance=0.3)  # 0.5 full extension of collar
    # between two adjacent segments
    dover_lap.write_rttm(
        os.path.join(out_dir, session_id + ".rttm"),
        output,
        channel=1,
    )
    rttm2json(os.path.join(out_dir, session_id + ".rttm"))


def diarize_audio(pipeline, wav_file, uem_boundaries=None):
    audio, fs = sf.read(wav_file, dtype="float32")
    if uem_boundaries is not None:
        uem_boundaries = [round(x * fs) for x in uem_boundaries]
    else:
        uem_boundaries = [0, len(audio)]
    assert audio.ndim == 1, "Multi-channel audio not supported right now."
    # cut out portions outside uem
    audio = audio[uem_boundaries[0] : uem_boundaries[1]]
    audio = torch.from_numpy(audio[None, ...])
    if IS_CUDA:
        audio = audio.cuda()

    # max speakers < 4. we can use this
    #  per rules here https://www.chimechallenge.org/current/task1/index
    result = pipeline.apply({"waveform": audio, "sample_rate": fs}, max_speakers=4)
    # result is an annotation, put back uem offset
    speakers = result.labels()
    offset = uem_boundaries[0] / fs
    new_annotation = Annotation()  # new annotation
    for spk in speakers:
        for seg in result.label_timeline(spk):
            new_annotation[Segment(seg.start + offset, seg.end + offset)] = spk

    return new_annotation


def read_uem(uem_file):
    with open(uem_file, "r") as f:
        lines = f.readlines()
    lines = [x.rstrip("\n") for x in lines]
    uem2sess = {}
    for x in lines:
        sess_id, _, start, stop = x.split(" ")
        uem2sess[sess_id] = (float(start), float(stop))
    return uem2sess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script performs diarization using Pyannote audio diarization pipeline "
        "plus dover-lap system combination across multiple microphones.",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-i,--in_dir",
        type=str,
        help="Folder containing the audio files which will be fed to the "
        "diarization pipeline.",
        metavar="STR",
        dest="in_dir",
    )
    parser.add_argument(
        "-o,--out_folder",
        type=str,
        default="",
        required=False,
        help="Path to output folder.",
        metavar="STR",
        dest="out_dir",
    )
    parser.add_argument(
        "-u,--uem",
        type=str,
        default="",
        required=False,
        help="Path to uem file.",
        metavar="STR",
        dest="uem_file",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Access token for HuggingFace Pyannote model."
        "see https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb",
        metavar="STR",
        dest="token",
    )
    parser.add_argument(
        "--mic_regex",
        type=str,
        help="Regular expression to extract the microphone channel from audio filename.",
        metavar="STR",
        dest="mic_regex",
    )
    parser.add_argument(
        "--sess_regex",
        type=str,
        help="Regular expression to extract the session from audio filename.",
        metavar="STR",
        dest="sess_regex",
    )

    args = parser.parse_args()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=args.token
    )

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    audio_f = glob.glob(os.path.join(args.in_dir, "*.wav")) + glob.glob(
        os.path.join(args.in_dir, "*.flac")
    )
    audio_f = [x for x in audio_f if re.search(args.mic_regex, Path(x).stem)]

    if args.uem_file:
        uem_map = read_uem(args.uem_file)

    sess2rttm = {}
    for audio_file in tqdm.tqdm(audio_f):
        filename = Path(audio_file).stem
        sess_name = re.search(args.sess_regex, filename).group()
        if args.uem_file:
            c_uem = uem_map[sess_name]
        else:
            c_uem = None
        # c_result = diarize_audio(pipeline, audio_file, c_uem)
        c_rttm_out = os.path.join(args.out_dir, Path(audio_file).stem + ".rttm")
        # with open(c_rttm_out, "w") as f:
        #   f.write(c_result.to_rttm())

        if sess_name not in sess2rttm.keys():
            sess2rttm[sess_name] = []
        sess2rttm[sess_name].append(c_rttm_out)

    for sess in sess2rttm.keys():
        c_sess_rttms = sess2rttm[sess]
        ensemble_doverlap(sess, c_sess_rttms, args.out_dir)
