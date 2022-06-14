import os
import numpy as np

from tqdm import tqdm
# from align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
# from IPython.display import HTML
# from base64 import b64encode
import argparse
import logging
import shutil

try:
    import dlib, cv2, skvideo
    import skvideo.io
    from skimage import transform as tf
    is_vis_preprocess_pkgs_avail = True
except ImportError:
    is_vis_preprocess_pkgs_avail = False


###########################################################

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## Based on: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/crop_mouth_from_video.py

""" Crop Mouth ROIs from videos for lipreading"""

import os,pickle,shutil,tempfile
import math
import subprocess
from collections import deque

# import os,pickle,shutil,tempfile
# import math
# import cv2
# import glob
# import subprocess
# import argparse
# import numpy as np
# from collections import deque
# import cv2
# from skimage import transform as tf
# from tqdm import tqdm

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):                                                 
        ret, frame = cap.read() # BGR
        if ret:                      
            yield frame                                                    
        else:                                                              
            break                                                         
    cap.release()

# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return

def crop_patch(video_pathname, landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin, start_idx, stop_idx, crop_height, crop_width):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        crop_height//2,
                                        crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[start_idx:stop_idx],
                                            crop_height//2,
                                            crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

###########################################################


def get_parser():
    """Returns the Parser object required to take inputs to data_prep.py"""
    parser = argparse.ArgumentParser(
        description="Parser for mouth roi extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_val_path", type=str, help="Path to the Train/ Validation files"
    )
    parser.add_argument("--test_path", type=str, help="Path to the Test files")
    parser.add_argument(
        "--face_predictor_path", type = str, help="Path to face predictor model"
    )
    parser.add_argument(
        "--mean_face_path", type = str, help = "Path to mean face file"
    )
    parser.add_argument(
        "--ffmpeg_path", type = str, help = "Path to ffmpeg"
    )
    
    return parser

    
def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68,2), dtype = np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path, ffmpeg_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in frames:
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if(preprocessed_landmarks is None):
        # 60% center crop the original video
        crop_video(input_video_path, output_video_path)
        return False
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, ffmpeg_path)
    return True

def crop_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x, y, w, h = int(w_frame * 0.2), int(h_frame * 0.2), int(w_frame * 0.6), int(h_frame * 0.6)

    crop_writer = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, crop_writer, fps, (w, h))
    success, frame = cap.read()
    if not success:
        print(">>> {} Cannot read mp4 file: vision features unavailable".format(input_video_path))
        # Final solution - copy original file video
        shutil.copyfile(input_video_path, output_video_path)
    while success:       
        crop_frame = frame[y:y+h, x:x+w]
        out.write(crop_frame)
        success, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = get_parser()
    args = parser.parse_args()

    if not is_vis_preprocess_pkgs_avail:
        raise ImportError(
            " You must have'dlib, cv2, skimage, skvideo' available. Please install it via" 
            " 'pip install dlib==19.17.0'"
            " 'pip install opencv-python'"
            " 'pip install sk-video'"
            " 'pip install scikit-image'"
            "  or 'cd path/'to/espnet/tools && . ./activate_python.sh"
            " && ./installers/install_vision_preprocess.sh ."
        )
    
    train_val_path = args.train_val_path
    test_path = args.test_path
    face_predictor_path = args.face_predictor_path
    mean_face_path = args.mean_face_path
    ffmpeg_path = args.ffmpeg_path

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    logging.info(f"Performing Data Preparation for TRAIN_VAL")
    count = 0
    file_count = 0
    for root, dirs, files in os.walk(train_val_path):
        for f in files:
            fname = os.path.splitext(f)
            if f.endswith(".mp4"):
                if fname[0].endswith("mouth_roi"):
                    continue
                elif os.path.exists(os.path.join(root, fname[0] + "_mouth_roi" + fname[1])):
                    continue
                else:
                    input_path = os.path.join(root, f)
                    output_path = os.path.join(root, fname[0] + "_mouth_roi" + fname[1])
                    if not preprocess_video(input_path, output_path, face_predictor_path, mean_face_path, ffmpeg_path):
                        count += 1
                    file_count += 1
            else:
                continue

    landmark_status = "Failed to detect landmarks for {} / {} Train videos".format(count, file_count)
    logging.info(landmark_status)

    logging.info(f"Performing Data Preparation for TEST")
    for root, dirs, files in os.walk(test_path):
        for f in files:
            fname = os.path.splitext(f)
            if f.endswith(".mp4"):
                if fname[0].endswith("mouth_roi"):
                    continue
                elif os.path.exists(os.path.join(root, fname[0] + "_mouth_roi" + fname[1])):
                    continue
                else:
                    input_path = os.path.join(root, f)
                    output_path = os.path.join(root, fname[0] + "_mouth_roi" + fname[1])
                    preprocess_video(input_path, output_path, face_predictor_path, mean_face_path, ffmpeg_path)
            else:
                continue

if __name__ == "__main__":
    main()