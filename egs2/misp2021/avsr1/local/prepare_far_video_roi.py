#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import codecs
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


def crop_frame_roi(frame, roi_bound, roi_size=(96, 96)):
    bound_l = max(roi_bound[3] - roi_bound[1], roi_bound[2] - roi_bound[0])
    bound_h_extend = (bound_l - roi_bound[2] + roi_bound[0]) / 2
    bound_w_extend = (bound_l - roi_bound[3] + roi_bound[1]) / 2
    x_start, x_end = int(roi_bound[1] - bound_w_extend), int(
        roi_bound[3] + bound_w_extend
    )
    if x_start < 0:
        x_start = 0
    if x_end > frame.shape[0]:
        x_end = frame.shape[0]
    y_start, y_end = int(roi_bound[0] - bound_h_extend), int(
        roi_bound[2] + bound_h_extend
    )
    if y_start < 0:
        y_start = 0
    if y_end > frame.shape[1]:
        y_end = frame.shape[1]
    roi_frame = frame[x_start:x_end, y_start:y_end, :]
    resized_roi_frame = cv2.resize(roi_frame, roi_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_roi_frame, cv2.COLOR_BGR2GRAY)


def crop_roi(frames_array, roi_bound, roi_size=(96, 96)):
    frames_num = frames_array.shape[0]
    assert frames_num == roi_bound.shape[0]
    roi_array = []
    for frame_idx in range(frames_num):
        roi_array.append(
            crop_frame_roi(
                frame=frames_array[frame_idx],
                roi_bound=roi_bound[frame_idx],
                roi_size=roi_size,
            )
        )
    return np.stack(roi_array, axis=0)


def segment_roi_json(
    roi_json_path,
    segments_name,
    segments_speaker,
    segments_start,
    segments_end,
    total_frames_num,
):
    with codecs.open(roi_json_path, "r") as handle:
        roi_dic = json.load(handle)

    def get_from_frame_detection(frame_i, target_id):
        if str(frame_i) in roi_dic:
            for roi_info in roi_dic[str(frame_i)]:
                if roi_info["id"] == target_id:
                    return [
                        roi_info["x1"],
                        roi_info["y1"],
                        roi_info["x2"],
                        roi_info["y2"],
                    ]
        return []

    delete_segments_key = []
    segments_roi_bound = {}
    for _, (name, speaker_id, frame_start, frame_end) in enumerate(
        zip(segments_name, segments_speaker, segments_start, segments_end)
    ):
        if frame_end >= total_frames_num:
            delete_segments_key.append(name)
            print(
                "{}: sengment end cross the line, {} but {}, skip".format(
                    name, frame_end, total_frames_num
                )
            )
        else:
            segment_roi_bound = []
            segment_roi_idx = []
            for frame_idx in range(frame_start, frame_end):
                segment_roi_bound.append(
                    get_from_frame_detection(frame_idx, speaker_id)
                )
                segment_roi_idx.append(frame_idx)

            frame_roi_exist_num = np.sum([*map(bool, segment_roi_bound)]).item()

            if float(frame_roi_exist_num) / float(frame_end - frame_start) < 0.5:
                delete_segments_key.append(name)
                print(
                    "{}: {}/{} frames have detection result, skip".format(
                        name, frame_roi_exist_num, frame_end - frame_start
                    )
                )
            elif frame_roi_exist_num == frame_end - frame_start:
                segments_roi_bound[name] = segment_roi_bound
                print(
                    "{}: {}/{} frames have detection result, prefect".format(
                        name, frame_roi_exist_num, frame_end - frame_start
                    )
                )
            else:
                print(
                    "{}: {}/{} frames have detection result, insert".format(
                        name, frame_roi_exist_num, frame_end - frame_start
                    )
                )
                i = 1
                forward_buffer = []
                forward_buffer_idx = -1
                while frame_start - i >= 0:
                    if get_from_frame_detection(frame_start - i, speaker_id):
                        forward_buffer = get_from_frame_detection(
                            frame_start - i, speaker_id
                        )
                        forward_buffer_idx = frame_start - i
                        break
                    else:
                        i += 1

                need_insert_idxes = []
                for i, (frame_idx, frame_roi_bound) in enumerate(
                    zip(segment_roi_idx, segment_roi_bound)
                ):
                    if frame_roi_bound:
                        while need_insert_idxes:
                            need_insert_idx = need_insert_idxes.pop(0)
                            if forward_buffer_idx == -1:
                                segment_roi_bound[need_insert_idx] = frame_roi_bound
                                print(
                                    need_insert_idx,
                                    segment_roi_bound[need_insert_idx],
                                    segment_roi_idx[need_insert_idx],
                                    frame_roi_bound,
                                    frame_idx,
                                )
                            else:
                                segment_roi_bound[need_insert_idx] = (
                                    (
                                        np.array(forward_buffer)
                                        + (
                                            segment_roi_idx[need_insert_idx]
                                            - forward_buffer_idx
                                        )
                                        * (
                                            np.array(frame_roi_bound)
                                            - np.array(forward_buffer)
                                        )
                                        / (frame_idx - forward_buffer_idx)
                                    )
                                    .astype(np.int64)
                                    .tolist()
                                )
                                print(
                                    need_insert_idx,
                                    segment_roi_bound[need_insert_idx],
                                    segment_roi_idx[need_insert_idx],
                                    frame_roi_bound,
                                    frame_idx,
                                    forward_buffer,
                                    forward_buffer_idx,
                                )
                        forward_buffer = frame_roi_bound
                        forward_buffer_idx = frame_idx
                    else:
                        need_insert_idxes.append(i)

                if need_insert_idxes:
                    i = 0
                    backward_buffer = []
                    backward_buffer_idx = -1
                    while frame_end + i < total_frames_num:
                        if get_from_frame_detection(frame_end + i, speaker_id):
                            backward_buffer = get_from_frame_detection(
                                frame_end + i, speaker_id
                            )
                            backward_buffer_idx = frame_end + i
                            break
                        else:
                            i += 1
                    while need_insert_idxes:
                        need_insert_idx = need_insert_idxes.pop(0)
                        if forward_buffer_idx == -1 and backward_buffer_idx == -1:
                            raise ValueError("no context cannot pad")
                        elif forward_buffer_idx == -1:
                            segment_roi_bound[need_insert_idx] = backward_buffer
                            print(
                                need_insert_idx,
                                segment_roi_bound[need_insert_idx],
                                segment_roi_idx[need_insert_idx],
                                backward_buffer,
                                backward_buffer_idx,
                            )
                        elif backward_buffer_idx == -1:
                            segment_roi_bound[need_insert_idx] = forward_buffer
                            print(
                                need_insert_idx,
                                segment_roi_bound[need_insert_idx],
                                segment_roi_idx[need_insert_idx],
                                forward_buffer,
                                forward_buffer_idx,
                            )
                        else:
                            segment_roi_bound[need_insert_idx] = (
                                (
                                    np.array(forward_buffer)
                                    + (
                                        segment_roi_idx[need_insert_idx]
                                        - forward_buffer_idx
                                    )
                                    * (
                                        np.array(backward_buffer)
                                        - np.array(forward_buffer)
                                    )
                                    / (backward_buffer_idx - forward_buffer_idx)
                                )
                                .astype(np.int64)
                                .tolist()
                            )
                            print(
                                need_insert_idx,
                                segment_roi_bound[need_insert_idx],
                                segment_roi_idx[need_insert_idx],
                                backward_buffer,
                                backward_buffer_idx,
                                forward_buffer,
                                forward_buffer_idx,
                            )
                assert not need_insert_idxes
                segments_roi_bound[name] = segment_roi_bound
    return segments_roi_bound, delete_segments_key


def segment_video_roi_json(
    video_path,
    roi_json_path,
    roi_store_dir,
    segments_name,
    segments_speaker,
    segments_start,
    segments_end,
    file_handle,
):
    segments_num = len(segments_start)
    assert segments_num > 0
    assert segments_num == len(segments_end)

    video_capture = cv2.VideoCapture(video_path)
    total_frames_num = int(video_capture.get(7))
    print(
        "using roi info from {}, all {} frames, generating {} segments".format(
            roi_json_path, total_frames_num, segments_num
        )
    )

    segments_roi_bound, delete_segments_key = segment_roi_json(
        roi_json_path,
        segments_name,
        segments_speaker,
        segments_start,
        segments_end,
        total_frames_num,
    )
    frame2segment_roi_bound = {}
    for i, segment_name in enumerate(segments_name):
        if segment_name not in delete_segments_key:
            segments_path = os.path.join(
                os.path.abspath(roi_store_dir), "{}.npz".format(segment_name)
            )
            file_handle.write("{} {}\n".format(segment_name, segments_path))
            if not os.path.exists(segments_path):
                for in_frame_idx in range(segments_end[i] - segments_start[i]):
                    if segments_start[i] + in_frame_idx in frame2segment_roi_bound:
                        frame2segment_roi_bound[
                            segments_start[i] + in_frame_idx
                        ].append(
                            [
                                segment_name,
                                in_frame_idx,
                                segments_end[i] - segments_start[i],
                                segments_roi_bound[segment_name][in_frame_idx],
                            ]
                        )
                    else:
                        frame2segment_roi_bound[segments_start[i] + in_frame_idx] = [
                            [
                                segment_name,
                                in_frame_idx,
                                segments_end[i] - segments_start[i],
                                segments_roi_bound[segment_name][in_frame_idx],
                            ]
                        ]

    if not os.path.exists(roi_store_dir):
        os.makedirs(roi_store_dir, exist_ok=True)

    if frame2segment_roi_bound:
        segments_roi_frames_buffer = {}

        # segment_video_writer = None
        frames_idx = 0

        # frames_bar = tqdm(total=total_frames_num, leave=True, desc='Frame')
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret and frame2segment_roi_bound:
                if frames_idx in frame2segment_roi_bound:
                    frame_info_list = frame2segment_roi_bound.pop(frames_idx)
                    for frame_info in frame_info_list:
                        if frame_info[1] == 0:
                            assert frame_info[0] not in segments_roi_frames_buffer
                            segments_roi_frames_buffer[frame_info[0]] = [
                                crop_frame_roi(frame, frame_info[3], (96, 96))
                            ]
                        else:
                            segments_roi_frames_buffer[frame_info[0]].append(
                                crop_frame_roi(frame, frame_info[3], (96, 96))
                            )

                        if frame_info[1] == frame_info[2] - 1:
                            np.savez(
                                os.path.join(
                                    roi_store_dir, "{}.npz".format(frame_info[0])
                                ),
                                data=segments_roi_frames_buffer.pop(frame_info[0]),
                            )
                frames_idx += 1
                # frames_bar.update(1)
            else:
                break
        # frames_bar.close()
        assert not frame2segment_roi_bound
        video_capture.release()
        print(
            "skip {} segments: {}".format(
                len(delete_segments_key), ",".join(delete_segments_key)
            )
        )
    return None


def input_interface(data_root, roi_json_dir):
    fps = 25

    video_dic = {}
    with codecs.open(os.path.join(data_root, "mp4.scp"), "r") as handle:
        lines_content = handle.readlines()
    for video_line in [*map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)]:
        name, path = video_line.split(" ")
        video_dic[name] = path

    vid2spk_dic = {}
    with codecs.open(os.path.join(data_root, "vid2spk"), "r") as handle:
        lines_content = handle.readlines()
    for vid2spk_line in [
        *map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)
    ]:
        name, speaker = vid2spk_line.split(" ")
        vid2spk_dic[name] = int(speaker[1:])

    segments_dic = {}
    with codecs.open(os.path.join(data_root, "segments"), "r") as handle:
        lines_content = handle.readlines()
    for segment_line in [
        *map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)
    ]:
        segment_name, name, start, end = segment_line.split(" ")
        if video_dic[name] not in segments_dic:
            segments_dic[video_dic[name]] = {
                "roi_json_path": os.path.join(roi_json_dir, "{}.json".format(name)),
                "segments_name": [segment_name],
                "segments_speaker": [vid2spk_dic[segment_name]],
                "segments_start": [int(np.around(float(start) * fps))],
                "segments_end": [int(np.around(float(end) * fps))],
            }
        else:
            segments_dic[video_dic[name]]["segments_name"].append(segment_name)
            segments_dic[video_dic[name]]["segments_speaker"].append(
                vid2spk_dic[segment_name]
            )
            segments_dic[video_dic[name]]["segments_start"].append(
                int(np.around(float(start) * fps))
            )
            segments_dic[video_dic[name]]["segments_end"].append(
                int(np.around(float(end) * fps))
            )

    return segments_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser("prepare_far_video_roi")
    parser.add_argument(
        "data_root",
        type=str,
        default="data/test_far_video",
        help="root directory of dataset",
    )
    parser.add_argument(
        "roi_json_dir", type=str, default="", help="directory of roi json"
    )
    parser.add_argument(
        "roi_store_dir",
        type=str,
        default="data/test_far_video",
        help="store directory of roi npz",
    )
    parser.add_argument("--ji", type=int, default=0, help="index of process")
    parser.add_argument("--nj", type=int, default=15, help="number of process")

    args = parser.parse_args()

    all_input_params = input_interface(
        data_root=args.data_root, roi_json_dir=args.roi_json_dir
    )
    all_sentences = sorted([*all_input_params.keys()])
    nj = args.nj
    ji = args.ji if nj > 1 else 0
    start_time = time.time()
    handle = codecs.open(
        os.path.join(args.roi_store_dir, "log", "roi.{}.scp".format(ji + 1)), "w"
    )

    for sentence_idx, sentence_path in enumerate(all_sentences):
        if sentence_idx % nj == ji:
            print("#" * 72)
            print("processing {}".format(sentence_path))
            segment_video_roi_json(
                video_path=sentence_path,
                roi_store_dir=args.roi_store_dir,
                file_handle=handle,
                **all_input_params[sentence_path]
            )
            in_len = (len(all_sentences) - ji) // nj
            in_index = (sentence_idx - ji) // nj
            current_dur = round((time.time() - start_time) / 60.0, 2)
            print(
                "{}/{} {}/{} min".format(
                    in_index,
                    in_len,
                    current_dur,
                    round(current_dur * (in_len + 1) / (in_index + 1), 2),
                )
            )

    handle.close()
