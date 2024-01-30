#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 KAIST (Minsu Kim)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import json
import os
import pickle
import re
import subprocess

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torchvision
from face_alignment import VideoProcess
from torchlm.models import pipnet
from tqdm import tqdm


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.include_wearer = True if args.include_wearer == "True" else False

    # face align and crop
    video_process = VideoProcess(
        mean_face_path="20words_mean_face.npy",
        convert_gray=False,
    )

    if not args.LRS3:
        val_session = ["Session_4", "Session_12"]
        test_session = ["Session_10", "Session_11"]

        file_time_lists = sorted(
            glob.glob(
                os.path.join(
                    args.data_dir, "Main", "Speech_Transcriptions", "*", "*.json"
                )
            )
        )

        # facial landmark detector
        landmark_detector = pipnet(
            backbone="resnet18",
            pretrained=True,
            num_nb=10,
            num_lms=98,
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            map_location="cuda",
            checkpoint=None,
        )

        for file in tqdm(file_time_lists):
            session, f_name = os.path.normpath(file).split(os.sep)[-2:]
            f_name = f_name[:-5]
            split = (
                "val"
                if session in val_session
                else "test" if session in test_session else "train"
            )

            face_json_file = open(
                os.path.join(
                    args.data_dir,
                    "Main",
                    "Face_Bounding_Boxes",
                    session,
                    f_name + ".json",
                )
            )
            face_data = json.load(face_json_file)
            face_BB = face_BB_extraction(face_data)

            with open(file, "rb") as json_file:
                data = json_file.read()
                data = data.decode("utf-8", errors="ignore")
                data = json.loads(data)
            data = sorted(data, key=lambda x: x["Start_Frame"])

            # delete face if the speaker doesn't say
            # a word during the session (eg, speaker 1)
            unique_sIDs = []
            for utt in data:
                unique_sIDs.append(utt["Participant_ID"])
            unique_sIDs = np.unique(unique_sIDs)
            orig_key = list(face_BB.keys())
            for key in orig_key:
                if key not in list(unique_sIDs):
                    del face_BB[key]

            wav_data, sample_rate = librosa.load(
                os.path.join(
                    args.data_dir,
                    "Main",
                    "Glasses_Microphone_Array_Audio",
                    session,
                    f_name + ".wav",
                ),
                sr=16000,
                mono=False,
            )
            # wav_data: 6 channels audio
            if args.beamforming:
                # to perform beamforming, please add the code lines here.
                assert NotImplementedError
            else:
                # use only 2nd channel (the center of glasses)
                wav_data = wav_data[1]

            video = load_video(
                os.path.join(
                    args.data_dir, "Main", "Video_Compressed", session, f_name + ".mp4"
                )
            )

            # Face video (Full Length)
            landmarks = landmark_detection(video, landmark_detector, face_BB)
            for face in landmarks:  # for each participant
                save_name = os.path.join(
                    f"./data/preprocess/{split}/Full_Video_Face/"
                    f"{session}_{f_name}_{face}.mp4"
                )
                if os.path.exists(save_name):
                    continue
                video_sequences, _, _ = video_process(video, landmarks[face])
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                save_video(save_name, video_sequences, face_BB[face], fps=20)

            max_len = len(wav_data)
            for utt in tqdm(data, leave=False):
                start_frame, end_frame, sID, text = (
                    utt["Start_Frame"],
                    utt["End_Frame"],
                    utt["Participant_ID"],
                    utt["Transcription"],
                )
                if not args.include_wearer and sID == 2:
                    continue

                text = text_normalize(text)
                if len(text) == 0:  # if empty text, pass the data
                    continue
                (
                    start_time,
                    end_time,
                    start_a_frame,
                    end_a_frame,
                ) = visual_frame_to_audio_frame(
                    start_frame, end_frame, max_len, extra_window=args.extra_window
                )
                segmented_wav_data = wav_data[start_a_frame:end_a_frame]

                # audio
                save_name = os.path.join(
                    f"./data/preprocess/{split}/Audio/"
                    f"{session}_{f_name}_{sID}_{start_frame}_{end_frame}.wav"
                )
                if not os.path.exists(save_name):
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)
                    sf.write(
                        save_name,
                        segmented_wav_data,
                        samplerate=sample_rate,
                        subtype="PCM_16",
                    )

                # text
                save_name = os.path.join(
                    f"./data/preprocess/{split}/Text/"
                    f"{session}_{f_name}_{sID}_{start_frame}_{end_frame}.txt"
                )
                if not os.path.exists(save_name):
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)
                    with open(save_name, "w") as txt:
                        txt.write(text)

                # video (Trimming) / sID is glasses wearer, so no face
                if sID != 2:
                    save_name = os.path.join(
                        f"./data/preprocess/{split}/Video/"
                        f"{session}_{f_name}_{sID}_{start_frame}_{end_frame}.mp4"
                    )
                    if not os.path.exists(save_name):
                        os.makedirs(os.path.dirname(save_name), exist_ok=True)
                        full_vid = (
                            f"./data/preprocess/{split}/Full_Video_Face/"
                            f"{session}_{f_name}_{sID}.mp4"
                        )
                        command = (
                            f"ffmpeg -loglevel panic -nostdin -y -i {full_vid}"
                            f" -ss {start_time} -to {end_time} -vcodec libx264"
                            f" -filter:v fps=fps=25 {save_name}"
                        )
                        subprocess.call(
                            command,
                            shell=True,
                        )
    else:
        pretrain_file_lists = sorted(
            glob.glob(
                os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True
            )
        )
        pretrain_file_lists = [
            f"{os.sep}".join(os.path.normpath(f).split(os.sep)[-3:])[:-4]
            for f in pretrain_file_lists
        ]
        train_file_lists = sorted(
            glob.glob(
                os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True
            )
        )
        train_file_lists = [
            f"{os.sep}".join(os.path.normpath(f).split(os.sep)[-3:])[:-4]
            for f in train_file_lists
        ]

        with open("local/lrs3-valid.id", "r") as txt:
            lines = txt.readlines()
        val_file_lists = [line.strip() for line in lines]
        train_file_lists = [f for f in train_file_lists if f not in val_file_lists]
        train_file_lists += pretrain_file_lists

        for mode, file_lists in tqdm(enumerate([train_file_lists, val_file_lists])):
            split = "train" if mode == 0 else "val"
            for file in tqdm(file_lists, leave=False):
                f_name = f"{file}"
                f_name_with_split = f"{os.sep}".join([split] + f_name.split(os.sep)[1:])
                file = os.path.join(args.data_dir, f_name + ".mp4")
                save_vid_name = os.path.join(
                    "./data/preprocess/LRS3/Video", f_name_with_split + ".mp4"
                )
                save_aud_name = os.path.join(
                    "./data/preprocess/LRS3/Audio", f_name_with_split + ".wav"
                )
                save_txt_name = os.path.join(
                    "./data/preprocess/LRS3/Text", f_name_with_split + ".txt"
                )
                if os.path.exists(os.path.join(args.landmark, f_name + ".pkl")):
                    with open(
                        os.path.join(args.landmark, f_name + ".pkl"), "rb"
                    ) as pkl_file:
                        landmarks = pickle.load(pkl_file)
                    video = load_video(file)

                    if all(x is None for x in landmarks) or len(video) == 0:
                        continue

                    if len(video) > 600:
                        continue

                    video_sequences, _, _ = video_process(video, landmarks)
                    os.makedirs(os.path.dirname(save_vid_name), exist_ok=True)
                    save_video(save_vid_name, video_sequences, None, fps=25)

                    os.makedirs(os.path.dirname(save_aud_name), exist_ok=True)
                    os.system(
                        f"ffmpeg -loglevel panic -nostdin -y -i {file} \
                            -acodec pcm_s16le -ar 16000 -ac 1 {save_aud_name}"
                    )

                    os.makedirs(os.path.dirname(save_txt_name), exist_ok=True)
                    with open(save_txt_name, "w") as txtw:
                        with open(
                            os.path.join(args.data_dir, f_name + ".txt"), "r"
                        ) as txt:
                            text = txt.readlines()[0][7:].strip()
                            text = text_normalize(text)
                            txtw.write(text)
    return


def load_video(data_filename):
    """load_video.

    :param data_filename: str, the filename for a video sequence.
    """
    frames = []
    cap = cv2.VideoCapture(data_filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)


def save_video(data_filename, video, bbs, fps):
    """save_video.

    :param data_filename: str, the filename for a video sequence.
    :param video: np.array, the video array.
    """
    if bbs is not None:
        # set black frame, if face is not detected
        mask = np.ones([video.shape[0], 1, 1, 1])
        for fr, bb in enumerate(bbs):
            if bb[0] == 0 and bb[2] == 0:
                mask[fr] = 0
        video = video * mask
    video = video[:, :, :, ::-1].copy()  # BGR -> RGB
    torchvision.io.write_video(
        filename=data_filename, video_array=torch.from_numpy(video), fps=fps
    )
    return


def visual_frame_to_audio_frame(start_frame, end_frame, max_len, extra_window=0):
    start_frame -= 1  # start from 1
    end_frame -= 1
    # video FPS: 20, audio FPS: 16000 (resampled)
    start_time = max((start_frame - extra_window) / 20, 0)
    start_a_frame = int(start_time * 16000)
    # video FPS: 20, audio FPS: 16000 (resampled)
    end_time = (end_frame + extra_window) / 20
    end_a_frame = min(int(end_time * 16000), max_len)
    return start_time, end_time, start_a_frame, end_a_frame


def text_normalize(text):
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)  # remove Coding conventions
    # remove punctuation except apostrophe
    text = re.sub(r"[^\w\s']|_", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()  # remove double space
    return text.lower()


def face_BB_extraction(face_data):
    # face bounding boxes for all participants;
    # if there is no face, the coordinates are set to zeros.
    unique_faces = []
    for data in face_data:
        faces = data["Participants"]
        if isinstance(faces, dict):
            unique_faces.append(faces["Participant_ID"])
        else:
            for face in faces:
                unique_faces.append(face["Participant_ID"])
    max_frame = face_data[-1]["Frame_Number"]
    unique_faces = np.unique(unique_faces)

    face_BB = {}
    for unique_face in unique_faces:
        face_BB[unique_face] = np.zeros([max_frame, 4], dtype=int)

    for fr_num, data in enumerate(face_data):
        assert data["Frame_Number"] - 1 == fr_num
        faces = data["Participants"]
        if isinstance(faces, dict):
            face_BB[faces["Participant_ID"]][fr_num] = [
                faces["x1"],
                faces["y1"],
                faces["x2"],
                faces["y2"],
            ]
        else:
            for face_num in range(len(faces)):
                face_BB[faces[face_num]["Participant_ID"]][fr_num] = [
                    faces[face_num]["x1"],
                    faces[face_num]["y1"],
                    faces[face_num]["x2"],
                    faces[face_num]["y2"],
                ]
    return face_BB


def landmark_detection(video, landmark_detector, face_BB):
    # 98 (WFLW) to 68 (ibug) landmark conversion
    ind68 = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        33,
        34,
        35,
        36,
        37,
        42,
        43,
        44,
        45,
        46,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        63,
        64,
        65,
        67,
        68,
        69,
        71,
        72,
        73,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
    ]
    lms = {}
    for speaker in face_BB:
        if speaker not in lms:
            lms[speaker] = []
        BBs = face_BB[speaker]
        for fr, bb in enumerate(BBs):
            if bb[0] == 0 and bb[1] == 0:
                lms[speaker].append(None)  # No Face is detected
            else:
                face = video[
                    fr, max(0, bb[1]) : bb[3], max(0, bb[0]) : bb[2], ::-1
                ]  # BGR -> RGB
                lm = landmark_detector.apply_detecting(face)
                lm[:, 0] += bb[0]
                lm[:, 1] += bb[1]  # 98, 2
                lms[speaker].append(lm[ind68])  # 68,2
    return lms


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for preprocessing."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="data_dir",
    )

    # for EASYCOM Dataset
    parser.add_argument(
        "--extra_window",
        type=int,
        default=3,
        help="extra frame for trimming the video/audio",
    )
    parser.add_argument(
        "--beamforming",
        default=False,
        help="Current version only supports False",
    )
    parser.add_argument(
        "--include_wearer",
        type=str,
        default="False",
    )

    # for LRS3 Dataset (additional Training data)
    parser.add_argument(
        "--LRS3",
        default=False,
        action="store_true",
        help="whether training with LRS3",
    )
    parser.add_argument(
        "--landmark",
        default=None,
        type=str,
        help="pre-extracted landmarks of LRS3",
    )
    return parser


if __name__ == "__main__":
    main()
