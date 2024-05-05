#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import kaldiio
import torch
import numpy as np

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.utils.types import str2bool


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_codec")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_choice", type=str, required=True)
    parser.add_argument("--codec_fs", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--dump_audio", type=str2bool, default=False)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--wav_wspecifier", type=str, default=None)
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser


class Codec_Tokenizer(object):
    def __init__(self, codec_choice, codec_fs, device, dump_audio=False):
        self.codec_choice = codec_choice
        self.device = device
        self.dump_audio = dump_audio

        if self.codec_choice == "DAC":
            try:
                import dac
            except:
                raise ImportError("Please install DAC with: pip install descript-audio-codec")

            model_path = dac.utils.download(
                model_type=str(codec_fs).replace("000", "khz")
            )
            self.codec = dac.DAC.load(model_path).to(device)
            self.n_codebook = self.codec.n_codebooks
            self.size_codebook = self.codec.codebook_size
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder_rates)
        
        elif self.codec_choice == "EnCodec":
            try:
                from encodec import EncodecModel
            except:
                raise ImportError("Please install Encodec with: pip install -U encodec")
            
            model_name = "encodec_model_" + str(codec_fs).replace("000", "khz")
            self.codec = getattr(EncodecModel, model_name)().to(device)
            bandwidth = 6.0 # 8 codebooks
            # bandwidth = max(self.codec.target_bandwidths) # 32 codebooks, too large
            self.codec.set_target_bandwidth(bandwidth)
            self.n_codebook = self.codec.quantizer.get_num_quantizers_for_bandwidth(
                self.codec.frame_rate, bandwidth
            )
            self.size_codebook = self.codec.quantizer.bins
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder.ratios)

        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def decode(self, codes):
        if self.codec_choice == "DAC":
            raise NotImplementedError
        elif self.codec_choice == "EnCodec":
            encoded_frames = [(codes.transpose(0, 1), None)]
            waveform = self.codec.decode(encoded_frames)
        else:
            raise NotImplementedError
        
        return waveform

    def __call__(self, wavs):
        # All wavs in shape of [batch_size, 1, num_samples]
        assert wavs.dim() == 3 and wavs.size(1) == 1

        if self.codec_choice == "DAC":
            z, codes = self.codec.encode(wavs)[:2]
            codes = codes.transpose(1, 2)
            if self.dump_audio:
                raise NotImplementedError
            else:
                resyn_audio = None

        elif self.codec_choice == "EnCodec":
            encoded_frames = self.codec.encode(wavs)
            codes = encoded_frames[0][0].transpose(1, 2)
            if self.dump_audio:
                resyn_audio = self.codec.decode(encoded_frames).squeeze(1)
            else:
                resyn_audio = None

        else:
            raise NotImplementedError

        # All codes in shape of [batch_size, T, n_codebook]
        # All resyn_audio in shape of [batch_size, num_samples]
        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)

        return codes, resyn_audio
    

def dump_codec(
    rspecifier,
    wspecifier,
    vocab_file,
    wav_wspecifier,
    codec_choice,
    codec_fs,
    batch_size,
    dump_audio,
    rank,
):
    # (1) Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logging.warning("Codec tokenization with CPU can be very slow.")
        logging.warning("Change batch_size=1 for CPU tokenization")
        args.batch_size = 1

    # (2) Codec model
    tokenizer = Codec_Tokenizer(codec_choice, codec_fs, device, dump_audio)

    # (3) Tokenizer loop
    codec_writer = kaldiio.WriteHelper(wspecifier)
    wav_reader = kaldiio.ReadHelper(rspecifier)
    if wav_wspecifier is not None and dump_audio:
        wav_ark_file, wav_scp_file = wav_wspecifier.split(":")[1].split(",")
        wav_scp_writer = open(wav_scp_file, "w")
        wav_ark_writer = open(wav_ark_file, "wb")
    else:
        wav_scp_writer, wav_ark_writer = None, None

    buffer, length_buffer, key_buffer = [], [], []
    wav_reader_len = len(open(rspecifier.split(":")[1]).readlines())
    for idx, (key, (sample_rate, wav)) in enumerate(wav_reader):
        if sample_rate != tokenizer.sample_rate:
            raise ValueError("Sample rate mismatch between input audio and codec model")

        if wav.ndim != 1:
            raise ValueError("Multi-Channel audio is not supported so far")

        wav = torch.from_numpy(wav)
        buffer.append(wav)
        length_buffer.append(len(wav))
        key_buffer.append(key)

        if idx == wav_reader_len - 1 or len(buffer) % batch_size == 0:
            wavs = pad_list(buffer, 0.0).to(device).unsqueeze(1).float()
            codes, resyn_wavs = tokenizer(wavs)

            codes = codes.detach().cpu().numpy()
            for code, length, key in zip(codes, length_buffer, key_buffer):
                code = code[: length // tokenizer.subsample * tokenizer.n_codebook]
                codec_writer[key] = code

            if dump_audio:
                resyn_wavs = resyn_wavs.detach().cpu().numpy()
                for wav, length, key in zip(resyn_wavs, length_buffer, key_buffer):
                    wav = wav[:length]
                    kaldiio.save_ark(
                        wav_ark_writer,
                        {key: (wav, sample_rate)},
                        scp=wav_scp_writer,
                        append=True,
                        write_function="soundfile",
                        write_kwargs={"format": "wav", "subtype": None},
                    )

            buffer, length_buffer, key_buffer = [], [], []

    # (4) dump vocabulary file
    if rank == 1:
        vocab_writer = open(vocab_file, "w")
        for codebook_idx in range(tokenizer.n_codebook):
            for code_idx in range(tokenizer.size_codebook):
                vocab_writer.write(f"<codec_layer{codebook_idx}_code{code_idx}>\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)
    dump_codec(**args)
