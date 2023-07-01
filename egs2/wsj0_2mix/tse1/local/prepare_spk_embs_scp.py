from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm.contrib.concurrent import thread_map


def compute_fbank(
    wav_path, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
):
    """Extract fbank.

    Simlilar to the one in wespeaker.dataset.processor,
    While integrating the wave reading and CMN.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat


def worker(uid_path, session, outdir):
    outdir = Path(outdir).absolute()

    uid, wav_path = uid_path
    feats = compute_fbank(wav_path)
    feats = feats.unsqueeze(0).numpy()  # add batch dimension
    embeddings = session.run(output_names=["embs"], input_feed={"feats": feats})
    key, value = Path(wav_path).stem, np.squeeze(embeddings[0])
    p = str(outdir / f"{key}.npy")
    np.save(p, value)
    return f"{uid} {p}\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scp", type=str, help="scp file containing paths to utterances")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the pretrained model in ONNX format",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of workers to process audio files in parallel",
    )
    parser.add_argument(
        "--max_chunksize",
        type=int,
        default=1000,
        help="Maximum size of chunks sent to worker processes",
    )
    args = parser.parse_args()

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(args.onnx_path, sess_options=so)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    tup = []
    with open(args.scp, "r") as f:
        for line in f:
            if not line.strip():
                continue
            uid, path = line.strip().split(maxsplit=1)
            tup.append((uid, path))
    # List[str]
    ret = thread_map(
        partial(worker, session=session, outdir=args.outdir),
        tup,
        max_workers=args.max_workers,
        chunksize=args.max_chunksize,
    )
    with open(f"{args.outdir}/embs.scp", "w") as f:
        for line in ret:
            f.write(line)
