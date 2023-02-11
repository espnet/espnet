import argparse
import os
from copy import deepcopy
from pathlib import Path

import lhotse
import soundfile as sf
import torch
import torchaudio
import tqdm
from torch.utils.data import DataLoader, Dataset


class EnvelopeVariance(torch.nn.Module):
    """
    Envelope Variance Channel Selection method with
    (optionally) learnable per mel-band weights.
    """

    def __init__(
        self,
        n_mels=40,
        n_fft=400,
        hop_length=200,
        samplerate=16000,
        eps=1e-6,
        chunk_size=4,
        chunk_stride=2,
    ):
        super(EnvelopeVariance, self).__init__()
        self.mels = torchaudio.transforms.MelSpectrogram(
            sample_rate=samplerate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2,
        )
        self.eps = eps
        self.subband_weights = torch.nn.Parameter(torch.ones(n_mels))
        self.chunk_size = int(chunk_size * samplerate / hop_length)
        self.chunk_stride = int(chunk_stride * samplerate / hop_length)

    def _single_window(self, mels):
        logmels = torch.log(mels + self.eps)
        mels = torch.exp(logmels - torch.mean(logmels, -1, keepdim=True))
        var = torch.var(mels ** (1 / 3), dim=-1)  # channels, subbands
        var = var / torch.amax(var, 1, keepdim=True)
        subband_weights = torch.abs(self.subband_weights)
        ranking = torch.sum(var * subband_weights, -1)
        return ranking

    def _count_chunks(self, inlen, chunk_size, chunk_stride):
        return int((inlen - chunk_size + chunk_stride) / chunk_stride)

    def _get_chunks_indx(self, in_len, chunk_size, chunk_stride, discard_last=False):
        i = -1
        for i in range(self._count_chunks(in_len, chunk_size, chunk_stride)):
            yield i * chunk_stride, i * chunk_stride + chunk_size
        if not discard_last and i * chunk_stride + chunk_size < in_len:
            if in_len - (i + 1) * chunk_stride > 0:
                yield (i + 1) * chunk_stride, in_len

    def forward(self, channels):
        assert channels.ndim == 3
        mels = self.mels(channels)
        if mels.shape[-1] > (self.chunk_size + self.chunk_stride):
            # using for because i am too lazy of taking care of padded
            # values in stats computation, but this is fast

            indxs = self._get_chunks_indx(
                mels.shape[-1], self.chunk_size, self.chunk_stride
            )
            all_win_ranks = [self._single_window(mels[..., s:t]) for s, t in indxs]

            return torch.stack(all_win_ranks).mean(0)
        else:
            return self._single_window(mels)


class MicRanking(Dataset):
    def __init__(self, recordings, supervisions, ranker, top_k):
        super().__init__()

        self.recordings = recordings
        self.supervisions = supervisions
        self.ranker = ranker
        self.top_k = top_k

    def __len__(self):
        return len(self.supervisions)

    def _get_read_chans(self, c_recordings, start, duration, fs=16000):
        to_tensor = []
        chan_indx = []
        for recording in c_recordings.sources:
            c_wav, _ = sf.read(
                recording.source,
                start=int(start * fs),
                stop=int(start * fs) + int(duration * fs),
            )
            c_wav = torch.from_numpy(c_wav).float().unsqueeze(0)
            assert (
                c_wav.shape[0] == 1
            ), "Input audio should be mono for channel selection in this script."

            if len(to_tensor) > 0:
                if c_wav.shape[-1] != to_tensor[0].shape[-1]:
                    print(
                        "Discarded {} because there is a difference of length of {}".format(
                            recording, c_wav.shape[-1] - to_tensor[0].shape[-1]
                        )
                    )
                    continue
            to_tensor.append(c_wav)

            chan_indx.append(recording.channels[0])

        all_channels = torch.stack(to_tensor).transpose(0, 1)

        return all_channels, chan_indx

    def __getitem__(self, item):
        c_supervision = self.supervisions[item]
        start = c_supervision.start
        duration = c_supervision.duration
        c_recordings = recordings[c_supervision.recording_id]
        fs = c_recordings.sampling_rate
        all_channels, chan_indx = self._get_read_chans(
            c_recordings, start, duration, fs
        )

        assert all_channels.ndim == 3
        assert (
            all_channels.shape[0] == 1
        ), "If batch size is more than one here something went wrong."
        with torch.inference_mode():
            c_scores = ranker(all_channels)
        c_scores = c_scores[0].numpy().tolist()
        c_scores = [(x, y) for x, y in zip(c_scores, chan_indx)]
        c_scores = sorted(c_scores, key=lambda x: x[0], reverse=True)
        c_scores = c_scores[: int(len(c_scores) * self.top_k)]
        new_sup = deepcopy(c_supervision)
        new_sup.channel = [x[-1] for x in c_scores]
        return new_sup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "We use this script to select a subset of " "microphones to feed to GSS."
    )
    parser.add_argument(
        "-r,--recordings",
        type=str,
        metavar="STR",
        dest="recordings",
        help="Input recordings lhotse manifest",
    )
    parser.add_argument(
        "-s,--supervisions",
        type=str,
        metavar="STR",
        dest="supervisions",
        help="Input supervisions lhotse manifest",
    )
    parser.add_argument(
        "-o, --out_name",
        type=str,
        metavar="STR",
        dest="out_name",
        help="Name and path for the new output manifests with the reduced "
        "channels. E.g. /tmp/chime6_selected --> will create "
        "chime6_selected_recordings.jsonl.gz  and chime6_selected_supervisions.jsonl.gz",
    )
    parser.add_argument(
        "-k, --top_k",
        default=25,
        type=int,
        metavar="INT",
        dest="top_k",
        help="Percentage of best microphones to keep (e.g. 20 -> 20% of all microphones)",
    )
    parser.add_argument(
        "--nj",
        default=8,
        type=int,
        metavar="INT",
        dest="nj",
        help="Number of parallel jobs",
    )
    args = parser.parse_args()

    recordings = lhotse.load_manifest(args.recordings)
    supervisions = lhotse.load_manifest(args.supervisions)
    output_filename = args.out_name
    ranker = EnvelopeVariance(samplerate=recordings[0].sampling_rate)
    single_thread = MicRanking(recordings, supervisions, ranker, args.top_k / 100)
    dataloader = DataLoader(
        single_thread,
        shuffle=False,
        batch_size=1,
        num_workers=args.nj,
        drop_last=False,
        collate_fn=lambda batch: [x for x in batch],
    )

    new_supervisions = []
    for i_batch, elem in enumerate(tqdm.tqdm(dataloader)):
        new_supervisions.extend(elem)

    recording_set, supervision_set = lhotse.fix_manifests(
        lhotse.RecordingSet.from_recordings(recordings),
        lhotse.SupervisionSet.from_segments(new_supervisions),
    )
    # Fix manifests
    lhotse.validate_recordings_and_supervisions(recording_set, supervision_set)

    Path(output_filename).parent.mkdir(exist_ok=True, parents=True)
    filename = Path(output_filename).stem
    supervision_set.to_file(
        os.path.join(Path(output_filename).parent, f"{filename}_supervisions.jsonl.gz")
    )
    recording_set.to_file(
        os.path.join(Path(output_filename).parent, f"{filename}_recordings.jsonl.gz")
    )
