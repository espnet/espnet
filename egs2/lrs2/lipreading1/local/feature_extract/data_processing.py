import skvideo.io
import skvideo.utils
import face_alignment
import numpy as np
import skimage.transform
import os
import sys


class BoundingBox(object):
    """
    A 2D bounding box
    """

    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = 255, 255
        self.maxx, self.maxy = 0, 0
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = int(x)
            if y < self.miny:
                self.miny = int(y)
            # Set max coords
            if x > self.maxx:
                self.maxx = int(x)
            if y > self.maxy:
                self.maxy = int(y)

    @property
    def width(self):
        return self.maxx - self.minx

    @property
    def height(self):
        return self.maxy - self.miny

    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)


# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cpu", flip_input=False)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


def video_face_crop(video_path="", output_path=""):
    video = skvideo.io.vread(video_path)

    preds = []

    for i in range(0, len(video), 3):
        pred = fa.get_landmarks(video[i])
        if pred:
            preds.append(pred[0])
    preds = np.array(preds)
    heatmap = np.median(preds, axis=0)

    bounding_box = BoundingBox(heatmap[2:15])

    croped = video[:,  bounding_box.miny:bounding_box.maxy,bounding_box.minx:bounding_box.maxx, :]

    crop_resize = np.zeros((np.shape(video)[0], 112, 112, np.shape(video)[-1]))

    for i in range(len(croped)):
        crop_resize[i] = skimage.transform.resize(croped[i], (112,112), preserve_range=True)

    crop_resize = crop_resize.astype(np.uint8)

    skvideo.io.vwrite(output_path, crop_resize)



def run_crop(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        new_root = root.replace(source_dir, target_dir)
        if files:
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            for file in files:
                if ".mp4" in file:
                    source_video = os.path.join(root, file)
                    target_video = os.path.join(new_root, file)
                    if os.path.exists(target_video):
                        print("SKIPING: ", source_video)
                        sys.stdout.flush()
                        continue
                    try:
                        video_face_crop(source_video, target_video)
                        print("DEALING: " + source_video)
                        sys.stdout.flush()
                    except Exception:
                        print("ERROR DEALING ", source_video, file=sys.stderr)
                        sys.stderr.flush()



def main():
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    print("source dir: ", source_dir)
    print("target dir: ", target_dir)
    run_crop(source_dir, target_dir)


if __name__ == "__main__":
    main()
