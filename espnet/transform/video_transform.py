import cv2
import dlib
from tqdm import tqdm
import numpy
from imutils import face_utils


def crop_frame(frame):
    return frame[480:1440, 270:810]

def load_video(filepath):
    video_stream = cv2.VideoCapture(filepath)
    rate = video_stream.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    print(rate)

    return rate, frames

def VideoReader(rspecifier, shape_predictor_path, size):
    reader = []
    lip_extractor = Lip_Extractor(shape_predictor_path)
    with open(rspecifier, 'r') as video_scp:
        for line in tqdm(video_scp.readlines()):
            line = line.strip()
            utt_id = line.split(' ')[0]
            video_path = line.split(' ')[1]
            rate, frames = load_video(video_path)
            print("Extracting lip features of {}...".format(utt_id))
            lip_frames = lip_extractor.catch_lip(frames, size)
            reader.append((utt_id, (rate, lip_frames)))
        return reader

class Lip_Extractor():
    def __init__(self, shape_predictor_path):
        self.face_detector = dlib.get_frontal_face_detector()
        self.lip_detector = dlib.shape_predictor(shape_predictor_path)
    def catch_lip(self, video_frames, size):
        lip_frames = []
        for i, video_frame in enumerate(video_frames):
            gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray_frame, 1)
            if len(faces) == 0:
                print("Could not find face in No. {} frame!".format(i))
                continue
            face = faces[0]
            shape = self.lip_detector(gray_frame, face)
            shape = face_utils.shape_to_np(shape)
            lip_indexs = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

            lip_shape = shape[lip_indexs[0]:lip_indexs[1]]

            (x, y, w, h) = cv2.boundingRect(numpy.array([lip_shape]))
            lip_frame = gray_frame[y:y+h, x:x+w]
            lip_frame = cv2.resize(lip_frame, size, interpolation=cv2.INTER_AREA)
            lip_frames.append(lip_frame)

        return(lip_frames)
    
            

