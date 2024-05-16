#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki, Wangyou Zhang, Chenda Li
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import librosa


import argparse
import fnmatch
import soundfile as sf


def predictor_setup(predictor_types, predictor_args, use_gpu=False):
    # Supported predictor types: utmos, dnsmos
    # Predictor args: predictor specific args
    predictor_dict = {}
    predictor_fs = {}
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if "aecmos" in predictor_types or "dnsmos" in predictor_types or "plcmos" in predictor_types:
        try:
            from speechmos import aecmos, dnsmos, plcmos
        except ImportError:
            raise ImportError("Please install speechmos for aecmos, dnsmos, and plcmos: pip install speechmos")
    
    for predictor in predictor_types:
        if predictor == "utmos":
            utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong").to(
                device
            )
            predictor_dict["utmos"] = utmos
            predictor_fs["utmos"] = 16000
        elif predictor == "dnsmos":
            predictor_dict["dnsmos"] = dnsmos
            predictor_fs["dnsmos"] = predictor_args["dnsmos"]["fs"]
        elif predictor = "aecmos":
            predictor_dict["aecmos"] = aecmos
            predictor_fs["aecmos"] = predictor_args["aecmos"]["fs"]
        elif predictor = "plcmos":
            predictor_dict["plcmos"] = plcmos
            predictor_fs["plcmos"] = predictor_args["plcmos"]["fs"]
        elif predictor == "pseq":
            try:
                from pesq import PesqError, pesq

                logging.warning("Using the PESQ package for evaluation")
            except ImportError:
                raise ImportError("Please install pesq and retry: pip install pesq")
        elif predictor == "stoi":
            try:
                from pystoi import stoi
            except ImportError:
                raise ImportError("Please install pystoi and retry: pip install stoi")
        else:
            raise NotImplementedError("Not supported {}".format(predictor))
    
    return predictor_dict, predictor_fs


def metrics(pred, fs, predictor_dict, predictor_fs, use_gpu=False):
    scores = {}
    for predictor in predictor_dict.keys():
        if predictor == "utmos":
            if fs != predictor_fs["utmos"]:
                pred_utmos = librosa.resample(pred, orig_sr=fs, target_sr=predictor_fs["utmos"])
            pred_tensor = torch.from_numpy(pred).unsqueeze(0)
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")
            score = predictor_dict["utmos"](pred_tensor, predictor_fs["utmos"])[0].item()
            scores.update(utmos=score)

        elif predictor == "aecmos":
            score = predictor["aecmos"].run(pred, sr=fs)
            scores.update(
                aec_echo_mos=score["echo_mos"],
                aec_deg_mos=score["deg_mos"],
            )
        elif predictor == "dnsmos":
            score = predictor["dnsmos"].run(pred, sr=fs)
            scores.update(
                dns_overall=score["ovrl_mos"],
                dns_p808=score["p808_mos"]
            )
        elif predictor == "plcmos":
            score = predictor["plcmos"].run(pred, sr=fs)
            scores.update(
                plcmos=score["plcmos"]
            )

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

    score_dict = {}
    for (gen_f, gt_f) in zip(gen_files, gt_files):
        pred_x, pred_fs = sf.read(gen_f, dtype="int16")
        gt_x, gt_fs = sf.read(gt_f, dtype="int16")

        if pred_fs != fs:
            pred_x = librosa.resample(pred_x.astype(np.float), pred_fs, fs)
        if gt_fs != fs:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, fs)
        
        scores = metrics(pred_x, gt_x, fs, f0min, f0max, dtw=False)
        gt_basename = _get_basename(gt_f)
        score_dict[gt_basename] = scores

    # write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)

    for metric in ["mcd", "f0rmse", "f0corr"]:
        with open(f"{args.outdir}/utt2{}".format(metric), "w") as f:
            for utt_id in sorted(score_dict.keys()):
                score = score_dict[utt_id][metric]
                f.write(f"{utt_id} {metric:.4f}\n")
        logging.info("Successfully finished {} evaluation.".format(metric))
    
    



    

    