import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
# from IPython.display import HTML
# from base64 import b64encode
import argparse
import logging
import shutil

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