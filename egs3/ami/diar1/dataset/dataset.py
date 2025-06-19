import numpy as np
from lhotse import CutSet
from torch.utils.data import Dataset
from lhotse.features import Fbank, FbankConfig
from lhotse.dataset.signal_transforms import SpecAugment
import torch
from collections import OrderedDict
import soundfile as sf



class DiarCutSet(Dataset):
    def __init__(self, data_dir="./data/ami", split=None,
                 is_training=False):
        if split is None:
            raise ValueError("This dataset requires a split name (e.g., 'dev', 'train')")
        path = f"{data_dir}/{split}-cuts.jsonl.gz"
        self.cuts = CutSet.from_file(path)
        self.is_training = is_training
        self.extractor = Fbank(FbankConfig(num_mel_bins = 23)) #FIXME

        self.specaugment = SpecAugment(time_warp_factor = 5,
        num_feature_masks = 2,
        features_mask_size = 4,
        num_frame_masks = 50,
        frames_mask_size = 10,
        max_frames_mask_fraction  = 0.2,
        p=0.9)


    def __len__(self):

        return len(self.cuts)


    def get_sa(self, supervisions, segment=120, max_spk=5, resolution=160, overlap_f=1, fs=16000):

        duration = segment * fs
        n_frames = int(np.floor((duration) / resolution))
        if len(supervisions) == 0:
            return torch.zeros((max_spk, n_frames), dtype=torch.bool)

        spk2indx = list(OrderedDict.fromkeys([x.speaker for x in supervisions]))
        spk2indx = {k: indx for indx, k in enumerate(spk2indx)}

        sa = torch.zeros((len(spk2indx.keys()), n_frames), dtype=torch.bool)
        for utt in supervisions:
            start = max(utt.start, 0.0)
            stop = min(utt.start + utt.duration, segment)
            start = int(start / (resolution / fs))
            stop = int(stop / (resolution / fs))
            sa[spk2indx[utt.speaker], start:stop] = 1

        if len(sa) < max_spk:
            sa = torch.nn.functional.pad(sa, (0, 0, 0, max_spk - len(sa)), mode="constant", value=0)
        return sa


    def _get_random_window(self, winlen, cut):

        start = np.random.uniform(cut.start, cut.duration+cut.start - winlen)
        stop = start + winlen
        return start, stop


    def _get_supervisions(self, cut, start, stop):

        return cut.truncate(offset=start,
                            duration=stop - start,
                            keep_excessive_supervisions=True)


    def read_audio(self, cut):

        sources = [x.source for x in cut.recording.sources]
        if self.is_training:
            source = np.random.choice(sources)
        else:
            source = sources[0]

        start = int(cut.start * cut.sampling_rate)
        stop = int((cut.start + cut.duration) * cut.sampling_rate)
        audio, _ = sf.read(source, start=start, stop=stop)
        if audio.ndim == 1:
            audio = audio[None, :]

        assert audio.shape[0] == 1
        return audio.astype(np.float32)

    def __getitem__(self, idx):

        cut = self.cuts[idx]
        supervisions = cut.supervisions
        audio = cut.load_features()
        mean = np.mean(audio, 0, keepdims=True)
        audio = audio - mean
        # specaugment
        if self.is_training:

            audio = self.specaugment(torch.from_numpy(audio[None, ...])).numpy()[0]
            #audio = audio + mean

        # need to get speaker activities here
        sa = self.get_sa(supervisions, cut.duration, fs=cut.sampling_rate).numpy().T

        example = {
            "speech": audio,
            "spk_labels": sa,
        }
        return example

