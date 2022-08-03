import collections.abc
import logging

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

try:
    import cv2

    is_cv2_avail = True
except ImportError:
    is_cv2_avail = False


class VisionFileReader(collections.abc.Mapping):
    """Reader class for 'vision'.

    Examples:
        key1 /some/path/a.mp4
        key2 /some/path/b.mp4
        key3 /some/path/c.mp4
        key4 /some/path/d.mp4
        ...

        >>> reader = VisionFileReader('vision.txt')
        >>> rate, array = reader['key1']
    """

    def __init__(
        self,
        fname,
    ):
        if not is_cv2_avail:
            raise ImportError(
                "'cv2' is not available. Please install it via "
                "'pip install opencv-python'"
                " or 'cd path/'to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_vision.sh ."
            )
        assert check_argument_types()
        self.fname = fname
        self.data = read_2column_text(fname)
        self.speed = 1.0

    def __getitem__(self, key):
        # Returns a cv2 video capture instance
        mp4 = self.data[key]
        vid = cv2.VideoCapture(mp4)
        rate = vid.get(cv2.CAP_PROP_FPS)  # fps
        return rate, vid

    def get_path(self, key):
        return self.data[key]

    def get_speed(self):
        return self.speed

    def __contains__(self, item):
        return item in self.data.keys()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class VisionDataset(collections.abc.Mapping):
    def __init__(self, loader, sample_step=1, normalize=True, dtype="float32"):
        assert check_argument_types()
        self.loader = loader
        self.dtype = dtype
        self.sample_step = sample_step
        self.sample_rate = 0.0
        self.vid_rate = 25.0
        self.normalize = normalize
        self.img_size = None
        self.init_sample_rate()
        logging.info(
            "Max Video Sample Rate Available : {} fps, "
            "Current Sampling Rate : {} fps".format(self.vid_rate, self.sample_rate)
        )

    def init_sample_rate(self):
        retval = self.loader[list(self.keys())[0]]
        assert len(retval) == 2, len(retval)
        rate, vidcap = retval
        self.vid_rate = rate
        self.sample_rate = rate / self.sample_step

    def keys(self):
        return self.loader.keys()

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, vsr):
        self.sample_rate = vsr
        if self.sample_step * vsr > self.vid_rate:
            self.sample_step = max(self.vid_rate // vsr, 1)
        self.vid_rate = self.sample_step * self.sample_rate
        return self.sample_step

    def set_sample_step(self, step: int):
        self.sample_step = step

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]
        assert len(retval) == 2, len(retval)
        rate, vid = retval  # Multichannel mp4 file
        if self.vid_rate != rate:
            vid.set(cv2.CAP_PROP_FPS, self.vid_rate)
        array = self.capture_video(vid)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        assert isinstance(array, np.ndarray), type(array)
        return array

    def capture_video(self, vidcap):
        """
        Captures video frames based on specified sample step

        Args:
            vidcap: cv2 video capture instance

        Output:
            data: numpy array of (T, H, W, C) containing image capture of video
        """
        data = []
        count = 0
        failure_count = 0
        prev_failure = False
        total_length = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        success = True
        success, image = vidcap.read()
        h, w, c = np.array(image).shape
        if self.img_size is None:
            self.img_size = h, w
        h, w = self.img_size
        while success:
            if count % self.sample_step == 0:
                if self.normalize:
                    image = cv2.normalize(
                        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
                    )
                image = cv2.resize(image, self.img_size)
                image = np.asarray(image)
                if (h, w, c) == image.shape:
                    data.append(image)
                    if prev_failure:
                        # Roll Back Count to the Correct Time-Stamp
                        count += failure_count
                        failure_count = 0
                        prev_failure = False
                else:
                    prev_failure = True
                    count = count - 1
                    failure_count += 1
                    if failure_count >= self.sample_step:
                        # move onto next sample capture
                        data.append(np.zeros(h, w, c))
                        count += failure_count
                        failure_count = 0
                        prev_failure = False
            success, image = vidcap.read()
            count += 1
        if len(data) == 0:
            data = np.zeros((total_length // self.sample_step + 1, h, w, c))
        return np.array(data)
