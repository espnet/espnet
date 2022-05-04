# The implementation of the metric for L3DAS22 in
# Guizzo. et al. "L3DAS22 Challenge: Learning 3D Audio
# Sources in a Real Office Environment"
# The code is based on:
# https://github.com/l3das/L3DAS22/blob/main/metrics.py


import argparse
import os
import sys
import warnings

import numpy as np
import soundfile as sf
import torchaudio
import torch
from tqdm import tqdm

import jiwer
import numpy as np
from pystoi import stoi
import transformers
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer


# TASK 1 METRICS
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")


def wer(clean_speech, denoised_speech):
    """
    computes the word error rate(WER) score for 1 single data point
    """

    def _transcription(clean_speech, denoised_speech):

        # transcribe clean audio
        input_values = wer_tokenizer(clean_speech, return_tensors="pt").input_values
        logits = wer_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_clean = wer_tokenizer.batch_decode(predicted_ids)[0]

        # transcribe
        input_values = wer_tokenizer(denoised_speech, return_tensors="pt").input_values
        logits = wer_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_estimate = wer_tokenizer.batch_decode(predicted_ids)[0]

        return [transcript_clean, transcript_estimate]

    transcript = _transcription(clean_speech, denoised_speech)
    try:  # if no words are predicted
        wer_val = jiwer.wer(transcript[0], transcript[1])
    except ValueError:
        wer_val = None

    return wer_val


def task1_metric(clean_speech, denoised_speech, sr=16000):
    """
    Compute evaluation metric for task 1 as (stoi+(1-word error rate)/2)
    This function computes such measure for 1 single datapoint
    """
    WER = wer(clean_speech, denoised_speech)
    if WER is not None:  # if there is no speech in the segment
        STOI = stoi(clean_speech, denoised_speech, sr, extended=False)
        WER = np.clip(WER, 0.0, 1.0)
        STOI = np.clip(STOI, 0.0, 1.0)
        metric = (STOI + (1.0 - WER)) / 2.0
    else:
        metric = None
        STOI = None
    return metric, WER, STOI


def main(args):
    # LOAD DATASET
    enh = []
    with open(args.predicted_path, "r") as f:
        for line in f.readlines():
            enh.append(line.split())
    ref = []
    with open(args.target_path, "r") as f:
        for line in f.readlines():
            ref.append(line.split())

    print("COMPUTING TASK 1 METRICS")
    print("M: Final Task 1 metric")
    print("W: Word Error Rate")
    print("S: Stoi")

    WER = 0.0
    STOI = 0.0
    METRIC = 0.0
    count = 0
    with tqdm(total=len(ref)) as pbar:
        for example_num, (key, ref_wav) in enumerate(ref):
            assert key == enh[example_num][0]
            target, sr = torchaudio.load(ref_wav)
            outputs, sr = torchaudio.load(enh[example_num][1])
            metric, wer, stoi = task1_metric(target.squeeze(0), outputs.squeeze(0), sr)
            if metric is not None:
                METRIC += (1.0 / float(example_num + 1)) * (metric - METRIC)
                WER += (1.0 / float(example_num + 1)) * (wer - WER)
                STOI += (1.0 / float(example_num + 1)) * (stoi - STOI)
            else:
                print("No voice activity on this frame")
            pbar.set_description(
                "M:"
                + str(np.round(METRIC, decimals=3))
                + ", W:"
                + str(np.round(WER, decimals=3))
                + ", S: "
                + str(np.round(STOI, decimals=3))
            )
            pbar.update(1)
            count += 1

    # print the results
    results = {"word error rate": WER, "stoi": STOI, "task 1 metric": METRIC}
    print("*******************************")
    print("RESULTS")
    for i in results:
        print(i, results[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--predicted_path", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")

    args = parser.parse_args()

    main(args)
