#!/usr/bin/env python3

# Copyright -2023 Takaaki Saeki, Wangyou Zhang, Chenda Li, Tomoki Hayashi
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import librosa
from discrete_speech_metrics import SpeechBERTScore, SpeechBLEU
from espnet2.bin.spk_inference import Speech2Embedding

import argparse
import fnmatch
import soundfile as sf


class SpeakerDistanceCalculator:
    """Extract speaker embedding and calculate cosine distance."""

    def __init__(
        self, model_tag=None, train_config=None, model_file=None, device="cpu"
    ):
        self.device = device
        if args["model_tag"] is not None:
            self.speech2spk_embned = Speech2Embedding.from_pretrained(
                model_tag=args["model_tag"]
            )
        else:
            assert (
                args["train_config"] is not None and args["model_file"] is not None
            ), "Cannot load model because no model file and model tag are given"

            self.speech2spk_embed = Speech2Embedding.from_pretrained(
                model_file=args["model_file"],
                train_config=args["train_config"],
            )
        self.speech2spk_embed = self.speech2spk_embed.to(device)

    def distance(self, pred, gt):
        """Calculate speaker embedding distance.

        Args:
            pred (np.array): prediction speech/audio. (Nsamples, )
            gt (np.array): ground truth speech/audio. (Nsamples, )

        Returns:
            float: cosine similarity of given speaker embedding
        """
        pred_embedding = self.speech2spk_embed(pred)
        gt_embedding = self.speech2spk_embed(gt)
        secs = 1 - spatial.distance.cosine(pred_embedding, gt_embedding)

        return secs


def predictor_setup(predictor_types, predictor_args, use_gpu=False):
    # Supported predictor types: utmos, dnsmos
    # Predictor args: predictor specific args
    predictor_dict = {}
    predictor_fs = {}

    device = "cuda" if use_gpu else "cpu"

    for predictor in predictor_types:
        if predictor == "speechbert":
            predictor_dict["speechbert"] = SpeechBERTScore(
                sr=16000, model_type="wavlm-large", layer=14, use_gpu=use_gpu
            )
            predictor_fs["speechbert"] = 16000
        elif predictor == "speechbleu":
            predictor_dict["speechbleu"] = SpeechBLEU(
                sr=16000,
                model_type="hubert-base",
                vocab=200,
                layer=11,
                n_ngram=2,
                remove_repetition=True,
                use_gpu=True,
            )
            predictor_fs["speechbleu"] = 16000
        elif predictor == "secs":
            predictor_dict["secs"] = SpeakerDistanceCalculator(
                model_tag=predictor_args["secs"]["model_tag"],
                model_file=predictor_args["secs"]["model_file"],
                train_config=predictor_args["secs"]["train_config"],
                device=device,
            )
            predictor_fs["secs"] = 16000
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return predictor_dict, predictor_fs


def _amp_normalization(audio):
    max_sample = np.amax(np.absolute(audio))
    return audio.astype(np.float32) / max_sample


def metrics(pred, gt, pred_fs, gt_fs, predictor_dict, predictor_fs, use_gpu=False):
    scores = {}
    for predictor in predictor_dict.keys():
        if predictor == "speechbleu":
            if pred_fs != predictor_fs["speechbleu"]:
                pred_speechbleu = librosa.resample(
                    pred, orig_sr=pred_fs, target_sr=predictor_fs["speechbleu"]
                )
            if gt_fs != predictor_fs["speechbleu"]:
                gt_speechbleu = librosa.resample(
                    pred, orig_sr=target_fs, target_sr=predictor_fs["speechbleu"]
                )
            pred_speechbleu = _amp_normalization(pred_speechbleu)
            gt_speechbleu = _amp_normalization(gt_speechbleu)
            spbleu = predictor_dict["speechbleu"](gt_speechbleu, pred_speechbleu)
            scores.update(speechbleu=spbleu)

        elif predictor == "speechbert":
            if pred_fs != predictor_fs["speechbert"]:
                pred_speechbert = librosa.resample(
                    pred, orig_sr=pred_fs, target_sr=predictor_fs["speechbert"]
                )
            if gt_fs != predictor_fs["speechbert"]:
                gt_speechbert = librosa.resample(
                    pred, orig_sr=target_fs, target_sr=predictor_fs["speechbert"]
                )
            pred_speechbert = _amp_normalization(pred_speechbert)
            gt_speechbert = _amp_normalization(gt_speechbert)
            spbs, _, _ = predictor_dict["speechbert"](gt_speechbert, pred_speechbert)
            scores.update(speechbert=spbs)

        elif predictor == "secs":
            if pred_fs != predictor_fs["secs"]:
                pred_secs = librosa.resample(
                    pred, orig_sr=pred_fs, target_sr=predictor_fs["secs"]
                )
            if gt_fs != predictor_fs["secs"]:
                gt_secs = librosa.resample(
                    pred, orig_sr=target_fs, target_sr=predictor_fs["secs"]
                )
            secs = predictor_dict["secs"](gt_secs, pred_secs)
            scores.update(secs=secs)
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return scores


def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> List[str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        List[str]: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Speech Resynthesis Performance."
    )
    parser.add_argument(
        "gen_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "gt_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run Speech Resynthesis Evaluation."""
    args = get_parser().parse_args()

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
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

    # find files
    if os.path.isdir(args.gen_wavdir_or_wavscp):
        gen_files = sorted(find_files(args.gen_wavdir_or_wavscp))
    else:
        with open(args.gen_wavdir_or_wavscp) as f:
            gen_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gen_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")
    if os.path.isdir(args.gt_wavdir_or_wavscp):
        gt_files = sorted(find_files(args.gt_wavdir_or_wavscp))
    else:
        with open(args.gt_wavdir_or_wavscp) as f:
            gt_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gt_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")

    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )
    logging.info("The number of utterances = %d" % len(gen_files))
