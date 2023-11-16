# From https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/generate_waveform_from_code.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import soundfile as sf
import torch
from fairseq import utils
from code_hifigan_vocoder import CodeHiFiGANVocoder

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_utt2units(path: str):
    ret = {}

    with open(path, encoding='utf-8') as f:
        for line in f:
            utt, seq = line.rstrip('\n').split(maxsplit=1)
            data = [int(e) for e in seq.split()]
            ret[utt] = data

    return ret


def main(args):
    logger.info(args)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.cpu
    logger.info(f"Use CUDA: ${use_cuda}")

    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg, cpu=not use_cuda)
    if use_cuda:
        vocoder = vocoder.cuda()

    multispkr = vocoder.model.multispkr
    if multispkr:
        logger.info("multi-speaker vocoder")
        num_speakers = vocoder_cfg.get(
            "num_speakers", 200
        )  # following the default in codehifigan to set to 200
        assert (
                args.speaker_id < num_speakers
        ), f"invalid --speaker-id ({args.speaker_id}) with total #speakers = {num_speakers}"

    data = read_utt2units(args.utt2units)

    with open(os.path.join(out_dir, "wav.scp"), "w", encoding='utf-8') as scp:
        for utt, units in data.items():
            x = dict(
                code=torch.LongTensor(units).view(1, -1),
            )

            # suffix = ""
            # if multispkr:
            #     spk = (
            #         random.randint(0, num_speakers - 1)
            #         if args.speaker_id == -1
            #         else args.speaker_id
            #     )
            #     suffix = f"_spk{spk}"
            #     x["spkr"] = torch.LongTensor([spk]).view(1, 1)

            x = utils.move_to_cuda(x) if use_cuda else x
            wav = vocoder(x, args.dur_prediction)

            wav_path = os.path.join(out_dir, f"{utt}.wav")
            sf.write(wav_path, wav.detach().cpu().numpy(), samplerate=16000)
            scp.write(f'{utt} {wav_path}\n')


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utt2units", type=str, required=True)
    parser.add_argument(
        "--vocoder", type=str, required=True, help="path to the CodeHiFiGAN vocoder"
    )
    parser.add_argument(
        "--vocoder_cfg",
        type=str,
        required=True,
        help="path to the CodeHiFiGAN vocoder config",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--dur_prediction",
        action="store_true",
        help="enable duration prediction (for reduced/unique code sequences)",
    )
    # parser.add_argument(
    #     "--speaker_id",
    #     type=int,
    #     default=-1,
    #     help="Speaker id (for vocoder that supports multispeaker). Set to -1 to randomly sample speakers.",
    # )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
