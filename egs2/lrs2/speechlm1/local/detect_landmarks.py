# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,os,pickle,math
import cv2,dlib,time
import numpy as np
from tqdm import tqdm
import skvideo
import skvideo.io
from kaldiio import WriteHelper

def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames

def detect_face_landmarks(face_predictor_path, cnn_detector_path, source_video_dir, landmark_dir, rank, nshard):

    def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path) 
    os.makedirs(landmark_dir, exist_ok=True)

    fids = []
    with open(source_video_dir, 'r') as file:
        for line in file.readlines():
            fids.append(line)
    
    rank -= 1
    num_per_shard = math.ceil(len(fids)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fids = fids[start_id: end_id]
    
    for fid in tqdm(fids):
        video_id, video_path = fid.strip().split(' ')
        output_fn = os.path.join(landmark_dir, video_id + '.pkl')
        # if os.path.exists(output_fn):
        #     continue
        import shutil
        history_path = '/nfs-02/yuyue/visualtts/reference_code/espnet/egs2/lrs2/avhubert/dump/video_feature_visual_tts_lrs2/landmark_dir/landmarks'
        if os.path.exists(os.path.join(history_path, video_id + '.pkl')):
            shutil.copy(os.path.join(history_path, video_id + '.pkl'), output_fn)
            continue
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_source_dir', type=str, default='/nfs-02/yuyue/visualtts/dataset/lrs2/video_25fps', help='root dir')
    parser.add_argument('--landmark_dir', default='/nfs-02/yuyue/visualtts/DSU-AVO-main/avhubert/preparation/data/lrs2/landmark', type=str, help='landmark dir')
    parser.add_argument('--cnn_detector', default='/nfs-02/yuyue/visualtts/reference_code/espnet/egs2/lrs2/avhubert/local/pretrained/cnn_detector.dat', type=str, help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', default='/nfs-02/yuyue/visualtts/reference_code/espnet/egs2/lrs2/avhubert/local/pretrained/face_predictor.dat', type=str, help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--rank', type=int, default=4, help='rank id')
    parser.add_argument('--nshard', type=int, default=5, help='number of shards')
    # parser.add_argument('--ffmpeg', type=str, default='/home/yuyue/wyy/anaconda3/envs/HPMDubbing/bin/ffmpeg', help='ffmpeg path')
    # parser.add_argument("--wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt")
    args = parser.parse_args()

    # import skvideo
    # skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    # print(skvideo.getFFmpegPath())
    # import skvideo.io
    detect_face_landmarks(args.face_predictor, args.cnn_detector, args.video_source_dir, args.landmark_dir, args.rank, args.nshard)
