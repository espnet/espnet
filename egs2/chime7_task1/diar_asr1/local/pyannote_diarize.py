import argparse
import glob
import json
import math
import os.path
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import tqdm
from dover_lap import dover_lap
from intervaltree import IntervalTree
from pyannote.audio import Pipeline
from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.signal import binarize
from pyannote.core import SlidingWindowFeature
from pyannote.metrics.segmentation import Annotation, Segment
from scipy.signal import medfilt2d
from copy import deepcopy

IS_CUDA = torch.cuda.is_available()
USE_DOVERLAP = False


def split_maxlen(utt_group, min_len=10):
    # merge if
    out = []
    stack = []
    for utt in utt_group:
        c_len = utt.end - utt.start
        if not stack or (c_len + sum([x.end - x.start for x in stack])) < min_len:
            stack.append(utt)
        else:
            out.append(Segment(stack[0].start, stack[-1].end))
            stack = [utt]
    return out


def merge_closer(annotation, delta=1.0, max_len=60, min_len=10):
    speakers = annotation.labels()
    new_annotation = Annotation()
    for spk in speakers:
        c_segments = sorted(annotation.label_timeline(spk), key= lambda x: x.start)
        stack = []
        for seg in c_segments:
            if not stack or abs(stack[-1].end - seg.start) < delta:
                stack.append(seg)
            else:
                # more than delta
                if sum([x.end - x.start for x in stack]) > max_len:
                    # break into parts of 10 seconds at least
                    for sub_seg in split_maxlen(stack, min_len):
                        new_annotation[sub_seg] = spk
                    stack = [seg]
                else:
                    new_annotation[Segment(stack[0].start, stack[-1].end)] = spk
                    stack = [seg]
    return new_annotation

def merge_closer_doverlap(turns, distance=0.5):
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
    output = merge_closer_doverlap(output, distance=0.5)
    # 0.5 full extension of collar between two adjacent segments
    dover_lap.write_rttm(
        os.path.join(out_dir, session_id + ".rttm"),
        output,
        channel=1,
    )
    rttm2json(os.path.join(out_dir, session_id + ".rttm"))


def diarize_session(pipeline, wav_files, uem_boundaries=None):
    # take the min len across all wavs
    minlen = min([sf.SoundFile(w).frames for w in wav_files])
    fs = sf.SoundFile(wav_files[0]).samplerate
    if uem_boundaries is not None:
        uem_boundaries = [round(x * fs) for x in uem_boundaries]
    else:
        uem_boundaries = [0, minlen]

    # now for each audio file run inference
    all_segmentation = []
    all_audio = []
    for w_f in tqdm.tqdm(wav_files):
        c_audio, c_fs = sf.read(w_f, dtype="float32")
        assert fs == c_fs
        c_audio = c_audio[: min(minlen, uem_boundaries[1])]
        c_audio = c_audio[uem_boundaries[0] :]
        c_audio = torch.from_numpy(c_audio[None, ...])
        if (c_audio**2).mean() < 1e-8:
            print(
                "Not running inference on {}, because the signal amplitude is "
                "too low, is it all zeros ?".format(c_audio)
            )
            continue
        if IS_CUDA:
            c_audio = c_audio.cuda()
        c_seg = pipeline.get_segmentations({"waveform": c_audio, "sample_rate": fs})
        c_seg = binarize(c_seg, onset=pipeline.segmentation.threshold,
                initial_state=False,
            )

        all_segmentation.append(c_seg)  # move to cpu for less mem consumption
        all_audio.append(c_audio)

    # here we select the best channel based on one with most activations.
    # not an optimal criterion but at least the clustering afterwards will be fast.
    sliding_window = all_segmentation[0].sliding_window
    all_audio = torch.cat(all_audio, 0)
    num_channels = all_audio.shape[0]
    num_chunks, frames, local_spk = all_segmentation[0].data.shape
    all_segmentation = SlidingWindowFeature(
        np.stack([x.data for x in all_segmentation], -1),
        sliding_window,
    )
    selected_audio = torch.zeros_like(c_audio)
    selected_seg = []
    for indx, (seg_b, segmentation) in enumerate(all_segmentation):
        c_seg = all_audio[:, math.floor(seg_b.start * fs) : math.floor(seg_b.end * fs)]
        # median filter here seems to improve performance on CHiME-6
        segmentation = medfilt2d(segmentation.reshape((frames, local_spk*num_channels)), (5, 1)).reshape((frames, local_spk, num_channels))
        selection = np.argmax(segmentation.sum((0, 1))) # not the best selection criteria
        # however this keeps it simple and fast.
        selected_audio[
            :, math.floor(seg_b.start * fs) : math.floor(seg_b.end * fs)
        ] = c_seg[selection]
        selected_seg.append(segmentation[..., selection])
    # stack em
    selected_seg = SlidingWindowFeature(
        np.stack([x.data for x in selected_seg]), sliding_window
    )
    count = Inference.trim(
        selected_seg, warm_up=(0.1, 0.1)
    )  # default value in Pyannote
    count = Inference.aggregate(
        np.sum(count, axis=-1, keepdims=True),
        frames=pipeline._frames,
        hamming=False,
        missing=0.0,
        skip_average=False,
    )
    count.data = np.rint(count.data).astype(np.uint8)
    embeddings = pipeline.get_embeddings(
        {"waveform": selected_audio, "sample_rate": fs},
        selected_seg,
        exclude_overlap=pipeline.embedding_exclude_overlap,
    )
    #  shape: (num_chunks, local_num_speakers, dimension)
    hard_clusters, _ = pipeline.clustering(
        embeddings=embeddings,
        segmentations=selected_seg,
        num_clusters=None,
        min_clusters=0,
        max_clusters=4,  # max-speakers are ok
        file={
            "waveform": selected_audio,
            "sample_rate": fs,
        },  # <== for oracle clustering
        frames=pipeline._frames,  # <== for oracle clustering
    )
    # reconstruct discrete diarization from raw hard clusters
    # keep track of inactive speakers
    inactive_speakers = np.sum(selected_seg.data, axis=1) == 0
    #  shape: (num_chunks, num_speakers)
    hard_clusters[inactive_speakers] = -2
    # reshape now to multi-channel
    discrete_diarization = pipeline.reconstruct(
        selected_seg,
        hard_clusters,
        count,
    )
    # convert to annotation
    result = pipeline.to_annotation(
        discrete_diarization,
        min_duration_on=0.0,
        min_duration_off=pipeline.segmentation.min_duration_off,
    )

    offset = uem_boundaries[0] / fs
    new_annotation = Annotation()  # new annotation
    speakers = result.labels()
    for spk in speakers:
        for seg in result.label_timeline(spk):
            new_annotation[Segment(seg.start + offset, seg.end + offset)] = spk

    new_annotation = merge_closer(new_annotation, delta=1.0, max_len=60)
    return new_annotation


def diarize_single_audio(pipeline, wav_file, uem_boundaries=None):
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
    # per rules here https://www.chimechallenge.org/current/task1/index
    if (audio**2).mean() < 1e-8:
        print(
            "Not running inference on {}, because the signal amplitude is "
            "too low, is it all zeros ?".format(wav_file)
        )
        return False

    result = pipeline({"waveform": audio, "sample_rate": fs}, max_speakers=4)
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
        "This script performs diarization using "
        "Pyannote audio diarization pipeline "
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
        "see https://github.com/pyannote/pyannote-audio"
        "/blob/develop/tutorials/applying_a_pipeline.ipynb",
        metavar="STR",
        dest="token",
    )
    parser.add_argument(
        "--mic_regex",
        type=str,
        help="Regular expression to extract the microphone "
        "channel from audio filename.",
        metavar="STR",
        dest="mic_regex",
    )
    parser.add_argument(
        "--sess_regex",
        type=str,
        help="Regular expression to extract the session" " from audio filename.",
        metavar="STR",
        dest="sess_regex",
    )

    args = parser.parse_args()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=args.token,
    )

    pipeline.embedding_batch_size = 128
    pipeline.segmentation_batch_size = 128
    #pipeline.segmentation.threshold = 0.3
    pipeline.segmentation.min_duration_off = 0.0

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    audio_f = glob.glob(os.path.join(args.in_dir, "*.wav")) + glob.glob(
        os.path.join(args.in_dir, "*.flac")
    )
    audio_f = [x for x in audio_f if re.search(args.mic_regex, Path(x).stem)]

    if args.uem_file:
        uem_map = read_uem(args.uem_file)

    if USE_DOVERLAP:
        sess2rttm = {}
        for audio_file in tqdm.tqdm(audio_f):
            filename = Path(audio_file).stem
            sess_name = re.search(args.sess_regex, filename).group()
            if args.uem_file:
                c_uem = uem_map[sess_name]
            else:
                c_uem = None
            c_result = diarize_single_audio(pipeline, audio_file, c_uem)
            if isinstance(c_result, bool) and c_result == False:
                continue
            c_rttm_out = os.path.join(args.out_dir, Path(audio_file).stem + ".rttm")
            with open(c_rttm_out, "w") as f:
                f.write(c_result.to_rttm())

            if sess_name not in sess2rttm.keys():
                sess2rttm[sess_name] = []
            sess2rttm[sess_name].append(c_rttm_out)

        for sess in sess2rttm.keys():
            c_sess_rttms = sess2rttm[sess]
            ensemble_doverlap(sess, c_sess_rttms, args.out_dir)
    else:
        # joint diarization of all mics
        sess2audio = {}
        for audio_file in audio_f:
            filename = Path(audio_file).stem
            sess_name = re.search(args.sess_regex, filename).group()
            if sess_name not in sess2audio.keys():
                sess2audio[sess_name] = []
            sess2audio[sess_name].append(audio_file)

        # now for each session
        for sess in sess2audio.keys():
            print("Diarizing Session {}".format(sess))
            if args.uem_file:
                c_uem = uem_map[sess]
            else:
                c_uem = None
            c_result = diarize_session(pipeline, sess2audio[sess], c_uem)
            c_rttm_out = os.path.join(args.out_dir, sess + ".rttm")
            with open(c_rttm_out, "w") as f:
                f.write(c_result.to_rttm())
            rttm2json(c_rttm_out)
