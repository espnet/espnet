import base64
import json
import os
import time
import traceback
from ast import literal_eval
from glob import glob
from pathlib import Path

import jiwer
import pandas as pd
import requests
from tacotron_cleaner.cleaners import *
from vocoder_eval import eval_rmse_f0

GROUND_TRUTH_DIR = "/home/perry/PycharmProjects/LJSpeech-1.1/wavs"

SPEECHMATICS_POST_URL = "https://asr.api.speechmatics.com/v2/jobs/"
SPEECHMATICS_GET_URL = "https://asr.api.speechmatics.com/v2/jobs/{job_id}/transcript"
SPEECHMATICS_FILE = "/home/perry/PycharmProjects/speechmatics_key.txt"
with open(SPEECHMATICS_FILE) as f:
    SPEECHMATICS_KEY = f.read().strip()
    SPEECHMATICS_HEADERS = {"Authorization": f"Bearer {SPEECHMATICS_KEY}"}


SAMPLE_RATE = 22050


def print_all_chars(split_csvs):
    charset = set()
    for split_csv in split_csvs:
        for chars in pd.read_csv(split_csv).raw_text:
            charset.update(set(chars))
    print("".sorted(charset))


PUNCS = re.compile(r'[!"\'(),-.:;?\[\]’“”]')


def normalize(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = expand_symbols(text)
    text = collapse_whitespace(text)
    return PUNCS.sub("", text)


def speechmatics_asr(speech_file):
    with open(speech_file, "rb") as f:
        files = {
            "data_file": f,
            "config": (
                None,
                '{"type": "transcription",'
                '"transcription_config": {"operating_point": "standard", "language": "en" }}',
            ),
        }
        response = requests.post(
            url=SPEECHMATICS_POST_URL, headers=SPEECHMATICS_HEADERS, files=files
        )
    if response.ok:
        job_id = literal_eval(response.text)["id"]
        kwargs = {
            "url": SPEECHMATICS_GET_URL.format(job_id=job_id),
            "params": {"format": "txt"},
            "headers": SPEECHMATICS_HEADERS,
        }
        transcript = requests.get(**kwargs)
        time.sleep(3)
        while not transcript.ok:
            if transcript.status_code == 404:
                time.sleep(1)
                transcript = requests.get(**kwargs)
            else:
                raise requests.HTTPError(transcript.text)
        return transcript.text
    else:
        raise requests.HTTPError(response.text)


def get_true_normalized_text(metadata_csv):
    import csv

    df = pd.read_csv(
        metadata_csv,
        sep="|",
        quoting=csv.QUOTE_NONE,
        names=["wav_name", "raw_text", "clean_text"],
    )
    normalized_text = df.clean_text.apply(normalize)
    return dict(zip(df.wav_name, normalized_text))


def compute_wer(transcripts_csv, wav_name_to_text, wav_suffix="_gen", verbose=False):
    wer_csv = Path(transcripts_csv).parent / "wer.csv"
    trans_df = pd.read_csv(transcripts_csv)

    results = []
    for wav_name, transcript in zip(trans_df.wav_name, trans_df.transcript):
        if wav_name.endswith(wav_suffix):
            wav_name = wav_name[: -len(wav_suffix)]
        true_text = wav_name_to_text[wav_name]
        wer = jiwer.wer(truth=true_text, hypothesis=transcript)
        words = len(true_text.split())
        wer_weighted = wer * words
        results.append([wav_name, wer, words, wer_weighted])
        if verbose:
            print(
                wav_name,
                f"True text: {true_text}",
                f"Transcript: {transcript}",
                wer,
                sep="\n",
            )
        else:
            print(wav_name, wer)
    results_df = pd.DataFrame(
        data=results, columns=["wav_name", "wer", "words", "wer_weighted"]
    )
    results_df.set_index("wav_name", inplace=True)
    results_df.sort_index(inplace=True)
    avg_wer = results_df.wer.mean()
    total_words = results_df.words.sum()
    avg_wer_weighted = results_df.wer_weighted.sum() / total_words
    results_df.loc["Total"] = avg_wer, total_words, avg_wer_weighted
    results_df.to_csv(wer_csv)
    print(f"Results saved in {wer_csv}")


def compute_f0_rmse(
    wav_dir, wav_suffix="_gen", ground_truth_dir=GROUND_TRUTH_DIR, verbose=False
):
    results_csv = os.path.join(wav_dir, "f0.csv")
    wav_files = glob(os.path.join(wav_dir, "*.wav"))
    assert wav_files, "No wav files found!"

    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        results = results_df.values.tolist()[:-1]
        existing_files = set(results_df.wav_name)
        wav_files = [f for f in wav_files if Path(f).stem not in existing_files]
    else:
        results = []
    try:
        for wav_file in wav_files:
            wav_name = Path(wav_file).stem
            if wav_suffix:
                wav_name = wav_name[: -len(wav_suffix)]
            gt_wav_file = os.path.join(ground_truth_dir, wav_name + ".wav")
            f0_rmse_mean, vuv_accuracy, vuv_precision = eval_rmse_f0(
                gt_wav_file, wav_file
            )
            results.append([wav_name, f0_rmse_mean, vuv_accuracy, vuv_precision])
            if verbose:
                print(
                    wav_name,
                    f"F0 RMSE: {f0_rmse_mean}",
                    f"Accuracy: {vuv_accuracy}",
                    f"Precision: {vuv_precision}",
                    sep="\n",
                )
            else:
                print(wav_name, f0_rmse_mean)
    except:
        traceback.print_exc()
    finally:
        results_df = pd.DataFrame(
            data=results,
            columns=["wav_name", "f0_rmse_mean", "vuv_accuracy", "vuv_precision"],
        )
        results_df.set_index("wav_name", inplace=True)
        results_df.sort_index(inplace=True)
        results_df.loc["Total"] = [
            results_df.f0_rmse_mean.mean(),
            results_df.vuv_accuracy.mean(),
            results_df.vuv_precision.mean(),
        ]
        results_df.to_csv(results_csv)
        print(f"Results saved in {results_csv}")


def transcribe_dir(wav_dir, asr_func):
    results_csv = os.path.join(wav_dir, "transcripts.csv")
    wav_files = glob(os.path.join(wav_dir, "*.wav"))
    assert wav_files, "No wav files found!"

    start_time = time.time()
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        results = results_df.values.tolist()
        existing_files = set(results_df.wav_name)
        wav_files = [f for f in wav_files if Path(f).stem not in existing_files]
        print(f"Existing: {len(existing_files)}, Remaining: {len(wav_files)}")
    else:
        results = []
    try:
        for wav_file in wav_files:
            wav_name = Path(wav_file).stem
            transcript = asr_func(wav_file)
            transcript_norm = normalize(transcript)
            results.append([wav_name, transcript_norm])
            print(wav_name, transcript_norm)
    except:
        traceback.print_exc()
    finally:
        results_df = pd.DataFrame(data=results, columns=["wav_name", "transcript"])
        results_df.set_index("wav_name", inplace=True)
        results_df.sort_index(inplace=True)
        results_df.to_csv(results_csv)
        print(f"Results saved in {results_csv}")
    total_time = int(time.time() - start_time)
    time_min = total_time // 60
    time_sec = total_time % 60
    print(f"Took {time_min} min {time_sec} sec")


def aggregrate_data(in_dirs, filename, out_dir=os.path.dirname(__file__)):
    assert out_dir not in in_dirs
    data = []
    for in_dir in in_dirs:
        full_path = os.path.join(in_dir, filename)
        if os.path.exists(full_path):
            with open(full_path) as f:
                lines = f.readlines()
                last_line = lines[-1]
                data.append([full_path] + last_line.split(",")[1:])
        else:
            print(f"Warning: {full_path} is missing")
            return
    columns = ["full_path"] + lines[0].split(",")[1:]
    df = pd.DataFrame(data=data, columns=columns)
    df.set_index("full_path", inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(out_dir, filename))


if __name__ == "__main__":
    exp_dirs = [
        "fastspeech2_varlr_40to0max75",
        "fastspeech2_varlr_40max75",
        "fastspeech2_varlr_20to0max75",
        "fastspeech2_varlr_20max75",
        "fastspeech2_0",
    ]

    wav_dirs = [
        f"/home/perry/PycharmProjects/espnet/egs2/ljspeech/tts1/exp/"
        f"{exp_dir}/decode_fastspeech_valid.loss.best/eval1_parallel_wavegan"
        for exp_dir in exp_dirs
    ]

    wav_name_to_text = get_true_normalized_text(
        "/home/perry/PycharmProjects/LJSpeech-1.1/metadata.csv"
    )
    for wav_dir in wav_dirs:
        print(wav_dir)
        transcribe_dir(wav_dir, asr_func=speechmatics_asr)
        compute_wer(wav_dir + "/transcripts.csv", wav_name_to_text, verbose=True)
        compute_f0_rmse(wav_dir)
    aggregrate_data(wav_dirs, filename="wer.csv")

    transcribe_dir(
        "/home/perry/PycharmProjects/LJSpeech-1.1/test_wavs", asr_func=speechmatics_asr
    )
    compute_wer(
        "/home/perry/PycharmProjects/LJSpeech-1.1/test_wavs/transcripts.csv",
        wav_name_to_text,
        verbose=True,
    )
