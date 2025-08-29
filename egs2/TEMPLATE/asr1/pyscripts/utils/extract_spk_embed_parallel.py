#!/usr/bin/env python3
#  2025, Terry(Zhuoyan) Tao

import argparse
import itertools
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import kaldiio
import librosa
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from espnet2.fileio.sound_scp import SoundScpReader

torch.backends.cudnn.benchmark = True


def get_parser():
    """Construct the parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() // 4),
        help="I/O+preproc worker threads (librosa resample, wav.scp reads).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,  # GPU-friendly default – adjust later
        help="Number of utterances processed together on the GPU",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=128,
        help="Max number of outstanding I/O tasks to prefetch.",
    )
    parser.add_argument("--pretrained_model", type=str, help="Pretrained model.")
    parser.add_argument(
        "--toolkit",
        type=str,
        help="Toolkit for Extracting speaker speaker embeddingss.",
        choices=["espnet", "speechbrain", "rawnet"],
    )
    parser.add_argument(
        "--spk_embed_tag",
        type=str,
        help="the target data name (e.g., xvector for xvector.scp)",
        default="spk_embed",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument(
        "in_folder", type=Path, help="Path to the input kaldi data directory."
    )
    parser.add_argument(
        "out_folder",
        type=Path,
        help="Output folder to save the speaker embeddings.",
    )
    return parser


class SpkEmbedExtractor:
    def __init__(self, args, device):
        self.toolkit = args.toolkit
        self.device = device
        self.tgt_sr = 16000  # NOTE(jiatong): following 16khz convetion

        if self.toolkit == "speechbrain":
            from speechbrain.dataio.preprocess import AudioNormalizer
            from speechbrain.pretrained import EncoderClassifier

            self.audio_norm = AudioNormalizer()
            self.model = EncoderClassifier.from_hparams(
                source=args.pretrained_model, run_opts={"device": device}
            )
        elif self.toolkit == "rawnet":
            from RawNet3 import RawNet3
            from RawNetBasicBlock import Bottle2neck

            self.model = RawNet3(
                Bottle2neck,
                model_scale=8,
                context=True,
                summed=True,
                encoder_type="ECA",
                nOut=256,
                out_bn=False,
                sinc_stride=10,
                log_sinc=True,
                norm_sinc="mean",
                grad_mult=1,
            )
            tools_dir = Path(os.getcwd()).parent.parent.parent / "tools"
            self.model.load_state_dict(
                torch.load(
                    tools_dir / "RawNet/python/RawNet3/models/weights/model.pt",
                    map_location=lambda storage, loc: storage,
                )["model"]
            )
            self.model.to(device).eval()
        elif self.toolkit == "espnet":
            from espnet2.bin.spk_inference import Speech2Embedding

            # NOTE(jiatong): set default config file as None
            # assume config is the same path as the model file
            speech2embedding_kwargs = dict(
                batch_size=128,
                dtype="float32",
                train_config=None,
                model_file=args.pretrained_model,
            )

            if args.pretrained_model.endswith("pth"):
                logging.info(
                    "the provided model path is end with pth,"
                    "assume it not a huggingface model"
                )
                model_tag = None
            else:
                logging.info(
                    "the provided model path is not end with pth,"
                    "assume use huggingface model"
                )
                model_tag = args.pretrained_model

            self.speech2embedding = Speech2Embedding.from_pretrained(
                model_tag=model_tag,
                **speech2embedding_kwargs,
            )
            self.speech2embedding.spk_model.to(device).eval()

    def _rawnet_extract_embd(self, audio, n_samples=48000, n_segments=10):
        if len(audio.shape) > 1:
            raise ValueError(
                "RawNet3 supports mono input only."
                f"Input data has a shape of {audio.shape}."
            )
        if len(audio) < n_samples:  # RawNet3 was trained using utterances of 3 seconds
            shortage = n_samples - len(audio) + 1
            audio = np.pad(audio, (0, shortage), "wrap")
        audios = []
        startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
        for asf in startframe:
            audios.append(audio[int(asf) : int(asf) + n_samples])
        audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32)).to(
            self.device
        )
        with torch.no_grad():
            output = self.model(audios)
        return output.mean(0).detach().cpu().numpy()

    def _espnet_extract_embd(self, audio):
        if len(audio.shape) == 2:
            logging.info(
                "Not support multi-channel input for ESPnet pre-trained model"
                f"Input data has shape {audio.shape}, default set avg across  channel"
            )
            audio = np.mean(audio, axis=0)
        elif len(audio.shape) > 1:
            raise ValueError(f"Input data has shape {audio.shape} thatis not support")
        audio = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        with torch.no_grad():
            output = self.speech2embedding(audio)
        return output.cpu().numpy()

    def __call__(self, wav, in_sr):
        if self.toolkit == "speechbrain":
            wav = self.audio_norm(torch.from_numpy(wav), in_sr).to(self.device)
            embeds = self.model.encode_batch(wav).detach().cpu().numpy()[0]
        elif self.toolkit == "rawnet":
            if in_sr != self.tgt_sr:
                wav = librosa.resample(wav, orig_sr=in_sr, target_sr=self.tgt_sr)
            embeds = self._rawnet_extract_embd(wav)
        elif self.toolkit == "espnet":
            if in_sr != self.tgt_sr:
                wav = librosa.resample(wav, orig_sr=in_sr, target_sr=self.tgt_sr)
            embeds = self._espnet_extract_embd(wav)
        return embeds

    def extract_batch(self, wav_list):
        """Batch version of __call__ for ESPnet/SpeechBrain/RawNet."""
        with torch.inference_mode():
            if self.toolkit == "espnet":
                # to float32 arrays
                arrs = []
                for w in wav_list:
                    if torch.is_tensor(w):
                        w = w.detach().cpu().numpy()
                    arrs.append(np.asarray(w, dtype=np.float32))

                # pad to (B, T)
                B = len(arrs)
                T = max(a.shape[0] for a in arrs) if B > 0 else 0
                batch_np = np.zeros((B, T), dtype=np.float32)
                for i, a in enumerate(arrs):
                    batch_np[i, : a.shape[0]] = a

                # -> torch (B, T) on the same device as the ESPnet model
                dev = next(self.speech2embedding.spk_model.parameters()).device
                speech = torch.from_numpy(batch_np).to(dev, non_blocking=True)

                # call the underlying speaker model directly
                out = self.speech2embedding.spk_model(
                    speech=speech,
                    spk_labels=None,
                    task_tokens=None,
                    extract_embd=True,  # return embeddings, not logits
                )

                if torch.is_tensor(out):
                    out = out.detach().cpu().numpy()
                return out.astype(np.float32)

            elif self.toolkit == "speechbrain":
                batch = torch.stack(
                    [
                        (
                            w
                            if torch.is_tensor(w)
                            else torch.as_tensor(w, dtype=torch.float32)
                        )
                        for w in wav_list
                    ]
                ).to(self.device)
                return (
                    self.model.encode_batch(batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            elif self.toolkit == "rawnet":
                outs = []
                for w in wav_list:
                    if torch.is_tensor(w):
                        w = w.detach().cpu().numpy()
                    outs.append(
                        self._rawnet_extract_embd(np.asarray(w, dtype=np.float32))
                    )
                return np.asarray(outs, dtype=np.float32)


def main(argv):
    """Load the model, generate kernel and bandpass plots."""
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    if torch.cuda.is_available() and ("cuda" in args.device):
        device = args.device
    else:
        device = "cpu"

    if args.toolkit in ("speechbrain", "rawnet", "espnet"):
        # Prepare spk2utt for mean x-vector
        spk2utt = dict()
        with open(os.path.join(args.in_folder, "spk2utt"), "r") as reader:
            for line in reader:
                details = line.split()
                spk2utt[details[0]] = details[1:]

        wav_scp = SoundScpReader(os.path.join(args.in_folder, "wav.scp"), np.float32)
        os.makedirs(args.out_folder, exist_ok=True)
        writer_utt = kaldiio.WriteHelper(
            "ark,scp:{0}/{1}.ark,{0}/{1}.scp".format(
                args.out_folder, args.spk_embed_tag
            )
        )
        writer_spk = kaldiio.WriteHelper(
            "ark,scp:{0}/spk_{1}.ark,{0}/spk_{1}.scp".format(
                args.out_folder, args.spk_embed_tag
            )
        )

        spk_embed_extractor = SpkEmbedExtractor(args, device)

        # Build flat work list (speaker, utt)
        work_iter = ((spk, utt) for spk, utts in spk2utt.items() for utt in utts)

        # Online stats per speaker: sum vector + count
        spk_sum = {}
        spk_cnt = {}
        _resamplers = {}
        _resampler_lock = mp.Lock()

        def load_item(spk_utt):
            spk, utt = spk_utt
            in_sr, wav = wav_scp[utt]  # wav: np.float32
            if args.toolkit in ("rawnet", "espnet"):
                tgt = spk_embed_extractor.tgt_sr
                if in_sr != tgt:
                    key = (in_sr, tgt)
                    if key not in _resamplers:
                        with _resampler_lock:
                            # Double-check to ensure thread safety
                            if key not in _resamplers:
                                _resamplers[key] = torchaudio.transforms.Resample(
                                    in_sr, tgt
                                )
                    w = torch.from_numpy(wav).float()
                    w = _resamplers[key](w.unsqueeze(0)).squeeze(0)  # torch 1-D
                    wav = w.numpy()
                    in_sr = tgt
            return spk, utt, in_sr, wav

        # ------------- Bounded prefetch loop WITH MINI-BATCHES -----------------
        batch = []  # holds (spk, utt, wav_tensor)
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            pending = set()
            for spk_utt in itertools.islice(work_iter, args.prefetch):
                pending.add(ex.submit(load_item, spk_utt))

            pbar = tqdm(total=sum(len(v) for v in spk2utt.values()))
            try:
                while pending:
                    fut = next(as_completed(pending))
                    pending.remove(fut)
                    speaker, utt, in_sr, wav = fut.result()

                    wav_tensor = torch.from_numpy(wav.astype(np.float32))
                    batch.append((speaker, utt, wav_tensor))

                    try:
                        nxt = next(work_iter)
                        pending.add(ex.submit(load_item, nxt))
                    except StopIteration:
                        pass

                    # flush when full or at the very end
                    # flush when full or at the very end
                    if len(batch) >= args.batch_size or (not pending and batch):
                        spks, utts, wav_tensors = zip(*batch)
                        embs = spk_embed_extractor.extract_batch(list(wav_tensors))
                        # embs is (B, D) np.float32 now
                        for spk, utt, emb in zip(spks, utts, embs):
                            writer_utt[utt] = emb
                            if spk not in spk_cnt:
                                spk_cnt[spk] = 1
                                spk_sum[spk] = emb.astype(np.float32, copy=True)
                            else:
                                spk_cnt[spk] += 1
                                spk_sum[spk] += emb
                        pbar.update(len(batch))
                        batch.clear()
            finally:
                pbar.close()
        # -----------------------------------------------------------------------

        # Write speaker means
        for spk, cnt in spk_cnt.items():
            writer_spk[spk] = spk_sum[spk] / max(1, cnt)
        writer_utt.close()
        writer_spk.close()

    else:
        raise ValueError(
            "Unkown type of toolkit. Only supported: speechbrain, rawnet, espnet, kaldi"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
