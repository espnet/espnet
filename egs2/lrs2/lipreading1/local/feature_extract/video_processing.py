import cvtransforms
import face_alignment
import numpy as np
import skimage.transform
import skvideo.io
import torch
from models import pretrained


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location="cpu")
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(pretrained_dict)
        print("load {} parameters".format(len(pretrained_dict)))
        model.load_state_dict(model_dict)
        return model


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
        if self.maxx <= self.minx or self.maxy <= self.miny:
            print("Box failed, return center box")
            self.minx, self.miny = 192, 192
            self.maxx, self.maxy = 64, 64

    @property
    def width(self):
        return self.maxx - self.minx

    @property
    def height(self):
        return self.maxy - self.miny

    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy
        )


def parse_scripts(scp_path, value_processor=lambda x: x, num_tokens=2):
    """
    Parse kaldi's script(.scp) file
    If num_tokens >= 2, function will check token number
    """
    scp_dict = dict()
    line = 0
    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if num_tokens >= 2 and len(scp_tokens) != num_tokens or len(scp_tokens) < 2:
                raise RuntimeError(
                    "For {}, format error in line[{:d}]: {}".format(
                        scp_path, line, raw_line
                    )
                )
            if num_tokens == 2:
                key, value = scp_tokens
            else:
                key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError(
                    "Duplicated key '{0}' exists in {1}".format(key, scp_path)
                )
            scp_dict[key] = value_processor(value)
    return scp_dict


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = pretrained.Lipreading(mode="temporalConv", nClasses=500)
model = reload_model(model, "./local/feature_extract/lipread_lrw_pretrain.pt")
model = model.float()
model.eval()
model.to(device)


class VideoReader(object):
    """
    Basic Reader Class
    """

    def __init__(self, scp_path, value_processor=lambda x: x):
        self.index_dict = parse_scripts(
            scp_path, value_processor=value_processor, num_tokens=2
        )
        self.index_keys = list(self.index_dict.keys())
        self.face_align_model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    def video_face_crop(self, input_video):
        video = input_video

        preds = []

        for i in range(0, len(video), 3):
            pred = self.face_align_model.get_landmarks(video[i])
            if pred:
                preds.append(pred[0])
        preds = np.array(preds)
        heatmap = np.median(preds, axis=0)

        bounding_box = BoundingBox(heatmap[2:15])

        croped = video[
            :,
            bounding_box.miny : bounding_box.maxy,
            bounding_box.minx : bounding_box.maxx,
            :,
        ]

        crop_resize = np.zeros((np.shape(video)[0], 112, 112, np.shape(video)[-1]))

        for i in range(len(croped)):
            try:
                crop_resize[i] = skimage.transform.resize(
                    croped[i], (112, 112), preserve_range=True
                )
            except Exception:
                print(croped)
                print("frame fails")

        crop_resize = crop_resize.astype(np.uint8)

        return crop_resize

    def transform_to_gray(self, data):
        r, g, b = data[..., 0], data[..., 1], data[..., 2]
        data = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
        return data

    def extract_feature(self, inputs):
        inputs = cvtransforms.ColorNormalize(inputs)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0).float()
        inputs = inputs.unsqueeze(1)
        with torch.no_grad():
            outputs = model(inputs.to(device))
        return outputs.cpu().numpy()

    def _load(self, key):
        # return path
        video = skvideo.io.vread(self.index_dict[key])
        v = self.video_face_crop(video)
        v = self.transform_to_gray(v)
        v = self.extract_feature(v)
        return v
        # return self.index_dict[key]

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # random index, support str/int as index
    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))
        if type(index) is int:
            # from int index to key
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError(
                    "Interger index out of range, {:d} vs {:d}".format(index, num_utts)
                )
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))
        return self._load(index)
