#!/usr/bin/env python3
#  2022, Hitachi LTD.; Nelson Yalta
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
from pathlib import Path

import kaldiio
import librosa
import numpy as np
import torch
from tqdm.contrib import tqdm

from espnet2.fileio.sound_scp import SoundScpReader


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
                batch_size=1,
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

        for speaker in tqdm(spk2utt):
            spk_embeddings = list()
            for utt in spk2utt[speaker]:
                in_sr, wav = wav_scp[utt]
                # Speaker Embedding
                embeds = spk_embed_extractor(wav, in_sr)
                writer_utt[utt] = np.squeeze(embeds)
                spk_embeddings.append(embeds)

            # Speaker Normalization
            embeds = np.mean(np.stack(spk_embeddings, 0), 0)
            writer_spk[speaker] = embeds
        writer_utt.close()
        writer_spk.close()

    else:
        raise ValueError(
            "Unkown type of toolkit. Only supported: speechbrain, rawnet, espnet, kaldi"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
