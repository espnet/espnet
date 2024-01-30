#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 KAIST (Minsu Kim)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile
from transforms import *

from espnet2.asr.encoder.avhubert_encoder import FairseqAVHubertEncoder
from espnet.utils.cli_writers import file_writer_helper

url = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/"
base_url = url + "noise-pretrain/base_vox_iter5.pt"
large_url = url + "noise-pretrain/large_vox_iter5.pt"


def build_file_list(file_list):
    with open(file_list, "r") as txt:
        lines = txt.readlines()
    return [line.strip().split() for line in lines]


def get_video_transform(split="test"):
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)

    return Compose(
        [
            Normalize(0.0, 255.0),
            CenterCrop(crop_size) if split == "test" else RandomCrop(crop_size),
            Identity() if split == "test" else HorizontalFlip(0.5),
            Normalize(mean, std),
            Identity() if split == "test" else TimeMask(max_mask_length=15),
            (
                Identity()
                if split == "test"
                else CutoutHole(min_hole_length=22, max_hole_length=44)
            ),
        ]
    )


def load_video(data_filename, grayscale=True):
    """load_video.

    :param filename: str, the fileanme for a video sequence.
    """
    frames = []
    cap = cv2.VideoCapture(data_filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)


def stacker(feats, stack_order):
    """
    Concatenating consecutive audio (100 fps) frames
    To match with video frame rate (25 fps),
    the 4 audio frames are concatenated in spectral dimension.
    T' = T / 4, F' = F * 4
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
        -1, stack_order * feat_dim
    )
    return feats


def per_file(file, video_transform, model, writer):
    id, file_name = file
    af_name = file_name[:-4]
    f_name = af_name.replace("Audio", "Video")
    # we can still extract features if only one modality is available
    # by using modality dropout property of AV-HuBERT
    if os.path.exists(f_name + ".mp4") or os.path.exists(af_name + ".wav"):
        if os.path.exists(f_name + ".mp4"):
            vid = load_video(f_name + ".mp4")

            if len(vid) == 0 or len(vid) > 600:
                return

            video_feats = video_transform(vid)
            if video_feats is None:
                return

        else:
            video_feats = None

        if os.path.exists(af_name + ".wav"):
            sample_rate, wav_data = wavfile.read(f"{af_name}.wav")
            audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(
                np.float32
            )  # [T, F]
            # [T/stack_order_audio, F*stack_order_audio]
            audio_feats = stacker(audio_feats, 4)
        else:
            audio_feats = None

        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate(
                    [
                        audio_feats,
                        np.zeros(
                            [-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype
                        ),
                    ]
                )
            elif diff > 0:
                audio_feats = audio_feats[:-diff]

        if audio_feats is not None:
            audio_feats = torch.FloatTensor(audio_feats)
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
            audio_feats = audio_feats.unsqueeze(0)
            # [B, T, F] -> [B, F, T]
            audio_feats = audio_feats.transpose(1, 2).contiguous()

        if video_feats is not None:
            # [B, T, H, W, C] -> [B, C, T, H, W]
            video_feats = (
                torch.FloatTensor(video_feats).unsqueeze(-1).unsqueeze(0)
            )  # B, T, H, W, C
            video_feats = video_feats.permute(0, 4, 1, 2, 3).contiguous()

        xs = {
            "video": video_feats.cuda() if video_feats is not None else None,
            "audio": audio_feats.cuda() if audio_feats is not None else None,
        }
        feats = model.forward_fusion(xs)

        writer[id] = feats.squeeze(0).transpose(0, 1).contiguous().cpu().numpy()
    return


def extract_noise_feature(file, model, save_name):
    if os.path.exists(file):
        video_feats = None

        sample_rate, wav_data = wavfile.read(file)
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(
            np.float32
        )  # [T, F]
        # [T/stack_order_audio, F*stack_order_audio]
        audio_feats = stacker(audio_feats, 4)

        audio_feats = torch.FloatTensor(audio_feats)
        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        audio_feats = audio_feats.unsqueeze(0)
        # [B, T, F] -> [B, F, T]
        audio_feats = audio_feats.transpose(1, 2).contiguous()

        xs = {
            "video": video_feats.cuda() if video_feats is not None else None,
            "audio": audio_feats.cuda() if audio_feats is not None else None,
        }
        feats = model.forward_fusion(xs)

        torch.save(feats.squeeze(0).transpose(0, 1).contiguous().cpu(), save_name)
    return


def main():
    parser = get_parser()
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu - 1)

    if args.model == "base":
        model = FairseqAVHubertEncoder(
            avhubert_url=base_url,
            avhubert_dir_path=args.pretrained_model_dir,
            encoder_embed_dim=768,
            encoder_layers=12,
            encoder_ffn_embed_dim=3072,
            encoder_attention_heads=12,
        )
    else:
        model = FairseqAVHubertEncoder(
            avhubert_url=large_url,
            avhubert_dir_path=args.pretrained_model_dir,
            encoder_embed_dim=1024,
            encoder_layers=24,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
        )

    model = model.cuda()
    model.eval()

    if args.noise_extraction:
        extract_noise_feature(args.noise_file, model, args.noise_save_name)
    else:
        writer = file_writer_helper(
            wspecifier=args.wspecifier,
            filetype="mat",
            write_num_frames=args.write_num_frames,
            compress=False,
        )

        video_transform = get_video_transform()

        file_lists = build_file_list(args.file_list)
        for kk, file in enumerate(file_lists):
            per_file(file, video_transform, model, writer)
            if kk % 10 == 0:
                print(f"{kk}/{len(file_lists)}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for preprocessing."
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default=None,
        help="file_list (scp)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="AV-HuBERT model config",
    )
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="./local/pre-trained",
        help="AV-HuBERT pretrained model path",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU number (job - 1)",
    )
    parser.add_argument(
        "--write_num_frames",
        type=str,
        default=None,
        help="specify wspecifer for num frames",
    )
    parser.add_argument(
        "--wspecifier",
        type=str,
        default=None,
        help="Write specifier",
    )
    parser.add_argument(
        "--noise_extraction",
        default=False,
        action="store_true",
        help="Whether extracting features of noise or data",
    )
    parser.add_argument(
        "--noise_file",
        type=str,
        default=None,
        help="noise file path",
    )
    parser.add_argument(
        "--noise_save_name",
        type=str,
        default="./data/babble_noise.pt",
        help="save path for noise feature",
    )
    return parser


if __name__ == "__main__":
    main()
