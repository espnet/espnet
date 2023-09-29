#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 KAIST (Minsu Kim)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse, cv2
import torch
import numpy as np
import pickle
from espnet2.asr.encoder.avhubert_encoder import FairseqAVHubertEncoder
from espnet.utils.cli_writers import file_writer_helper
from python_speech_features import logfbank
from scipy.io import wavfile
import torch.nn.functional as F
from face_alignment import VideoProcess

def build_file_list(file_list):
    with open(file_list, 'r') as txt:
        lines = txt.readlines()
    return [l.strip().split() for l in lines]

def load_video(data_filename):
    """load_video.

    :param filename: str, the fileanme for a video sequence.
    """
    frames = []
    cap = cv2.VideoCapture(data_filename)                           
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR                                            
        if ret:                                                                  
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)

def stacker(feats, stack_order):
    """
    Concatenating consecutive audio frames
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
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    return feats

def per_file(f, args, video_process, model, writer):
    id, f = f
    f_name = f'{os.sep}'.join(os.path.normpath(f).split(os.sep)[-3:])[:-4]
    temp_aud_path = os.path.join(f'./data/temp/{args.job}', f_name)
    if not os.path.exists(os.path.dirname(temp_aud_path)): os.makedirs(os.path.dirname(temp_aud_path), exist_ok=True)
    if os.path.exists(os.path.join(args.landmark_path, f_name + '.pkl')):
        with open(os.path.join(args.landmark_path, f_name + '.pkl'), "rb") as pkl_file:
            lm = pickle.load(pkl_file)
        vid_name = f
        vid = load_video(vid_name)

        if all(x is None for x in lm) or len(vid) == 0:
            return

        if len(vid) > 600:
            return

        output = video_process(vid, lm)
        if output is None:
            return

        os.system(f'ffmpeg -loglevel panic -nostdin -y -i {vid_name} -acodec pcm_s16le -ar 16000 -ac 1 {temp_aud_path}.wav')
        
        video_feats, _, _ = output  # T, H, W

        sample_rate, wav_data = wavfile.read(f'{temp_aud_path}.wav')
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
        audio_feats = stacker(audio_feats, 4) # [T/stack_order_audio, F*stack_order_audio]
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        audio_feats = torch.FloatTensor(audio_feats)
        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        audio_feats = audio_feats.unsqueeze(0)

        # [B, T, H, W, C] -> [B, C, T, H, W]
        video_feats = torch.FloatTensor(video_feats).unsqueeze(-1).unsqueeze(0) # B, T, H, W, C
        video_feats = video_feats.permute(0, 4, 1, 2, 3).contiguous()
        # [B, T, F] -> [B, F, T]
        audio_feats = audio_feats.transpose(1, 2).contiguous()

        xs = {'video': video_feats.cuda(), 'audio': audio_feats.cuda()}
        feats = model.forward_fusion(xs)
        
        writer[id] = feats.squeeze(0).transpose(0, 1).contiguous().cpu().numpy()

        os.system(f'rm {temp_aud_path}.wav')
    return

def main():
    parser = get_parser()
    args = parser.parse_args()
    torch.cuda.set_device(args.job - 1)
    output_dir = os.path.split(args.file_list)[0]
    output_ark = os.path.join(output_dir, f'feature.{args.job}.ark')
    output_scp = os.path.join(output_dir, f'feature.{args.job}.scp')
    writer = file_writer_helper(
        wspecifier=f"ark,scp:{output_ark},{output_scp}",
        filetype="mat",
        write_num_frames=f"ark,t:{output_dir}/num_frames.{args.job}.txt",
        compress=False,
    )

    if args.model == 'base':
        model = FairseqAVHubertEncoder(avhubert_url='https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/base_vox_iter5.pt',
                                    avhubert_dir_path=f'./local/pre-trained',
                                    encoder_embed_dim = 768,
                                    encoder_layers = 12,
                                    encoder_ffn_embed_dim = 3072,
                                    encoder_attention_heads = 12,
                                    )
    else:
        model = FairseqAVHubertEncoder(avhubert_url='https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt',
                            avhubert_dir_path=f'./local/pre-trained',
                            encoder_embed_dim = 1024,
                            encoder_layers = 24,
                            encoder_ffn_embed_dim = 4096,
                            encoder_attention_heads = 16,
                            )
    model = model.cuda()
    model.eval()
    video_process = VideoProcess(mean_face_path='20words_mean_face.npy', convert_gray=True)
    file_lists = build_file_list(args.file_list)
    for kk, f in enumerate(file_lists):
        per_file(f, args, video_process, model, writer)
        if kk % 10 == 0:
            print(f'{kk}/{len(file_lists)}')
    
def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for preprocessing."
    )
    parser.add_argument(
        "--file_list", type=str, required=True, help="file_list (scp)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="AV-HuBERT model config"
    )
    parser.add_argument(
        "--job", type=int, required=True, help="job number"
    )
    parser.add_argument(
        "--landmark_path", type=str, default='./local/LRS3_landmarks', help="path including landmark files"
    )
    return parser


if __name__ == "__main__":
    main()