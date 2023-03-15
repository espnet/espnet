"""
@author: chkarada
"""

# Adapted from DNS-Challenge official repo file:
# https://github.com/microsoft/DNS-Challenge/blob/master/noisyspeech_synthesizer_singleprocess.py

# Original License: CC-BY 4.0 International:
# https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE

# Retrieved on Sep. 7th, 2022, by Shih-Lun Wu (summer7sean@gmail.com)

import argparse
import configparser as CP
import glob
import multiprocessing
import os
import sys
from pathlib import Path
from random import shuffle

import librosa
import numpy as np
import pandas as pd
from audiolib import (
    activitydetector,
    audioread,
    audiowrite,
    is_clipped,
    segmental_snr_mixer,
)
from scipy import signal

import utils

MAXTRIES = 50
MAXFILELEN = 100


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]

    return reverb_speech


def build_audio(is_clean, params, index, audio_samples_length=-1):
    """Construct an audio signal from source files"""

    fs_output = params["fs"]
    silence_length = params["silence_length"]
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    if is_clean:
        source_files = params["cleanfilenames"]
        idx = index
    else:
        if "noisefilenames" in params.keys():
            source_files = params["noisefilenames"]
            idx = index
        # if noise files are organized into individual subdirectories,
        # pick a directory randomly
        else:
            noisedirs = params["noisedirs"]
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = glob.glob(
                os.path.join(noisedirs[idx_n_dir], params["audioformat"])
            )
            shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:
        # read next audio file and resample if necessary

        idx = (idx + 1) % np.size(source_files)
        input_audio, fs_input = audioread(source_files[idx])
        if input_audio is None:
            sys.stderr.write("WARNING: Cannot read file: %s\n" % source_files[idx])
            continue
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, fs_input, fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (
            not is_clean or not params["is_test_set"]
        ):
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg : idx_seg + remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0 and not is_clean and "noisedirs" in params.keys():
        print(
            "There are not enough non-clipped files in the "
            + noisedirs[idx_n_dir]
            + " directory to complete the audio build"
        )
        return [], [], clipped_files, idx

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    """Calls build_audio() to get an audio signal, and verify that it meets the
    activity threshold"""

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])
    if is_clean:
        activity_threshold = params["clean_activity_threshold"]
    else:
        activity_threshold = params["noise_activity_threshold"]

    while True:
        audio, source_files, new_clipped_files, index = build_audio(
            is_clean, params, index, audio_samples_length
        )

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, index


def main_gen_single_process(params, proc_idx, st_idx, ed_idx, total_procs):
    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    file_num = st_idx
    clean_index = round(len(params["cleanfilenames"]) / total_procs * proc_idx)
    noise_index = round(len(params["noisefilenames"]) / total_procs * proc_idx)

    while file_num < ed_idx:
        if (file_num - st_idx) % 10 == 9:
            print(
                "[INFO] (noisyspeech_synthesizer.py, \
                 process {}) generating wav mixture no. {} / {}".format(
                    proc_idx, file_num - st_idx + 1, ed_idx - st_idx
                ),
                file=sys.stderr,
            )
        # generate clean speech
        clean, clean_sf, clean_cf, clean_laf, clean_index = gen_audio(
            True, params, clean_index
        )

        # generate noise
        noise, noise_sf, noise_cf, noise_laf, noise_index = gen_audio(
            False, params, noise_index, len(clean)
        )

        clean_clipped_files += clean_cf
        clean_low_activity_files += clean_laf
        noise_clipped_files += noise_cf
        noise_low_activity_files += noise_laf

        # get rir files and config

        # mix clean speech and noise
        # if specified, use specified SNR value
        if not params["randomize_snr"]:
            snr = params["snr"]
        # use a randomly sampled SNR value between the specified bounds
        else:
            snr = np.random.randint(params["snr_lower"], params["snr_upper"])

        clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(
            params=params, clean=clean, noise=noise, snr=snr
        )
        # Uncomment the below lines if you need segmental SNR
        # and comment the above lines using snr_mixer
        # clean_snr, noise_snr, noisy_snr, target_level = \
        #      segmental_snr_mixer(params=params,
        #                           clean=clean,
        #                           noise=noise,
        #                           snr=snr)
        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            print(
                "Warning: File #"
                + str(file_num)
                + " has unexpected clipping, "
                + "returning without writing audio to disk"
            )
            continue

        clean_source_files += clean_sf
        noise_source_files += noise_sf

        # write resultant audio streams to files
        hyphen = "-"
        clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
        noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
        noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

        noisyfilename = (
            clean_files_joined
            + "_"
            + noise_files_joined
            + "_snr"
            + str(snr)
            + "_tl"
            + str(target_level)
            + "_fileid_"
            + str(file_num)
            + ".wav"
        )
        cleanfilename = "clean_fileid_" + str(file_num) + ".wav"
        noisefilename = "noise_fileid_" + str(file_num) + ".wav"

        noisypath = os.path.join(params["noisyspeech_dir"], noisyfilename)
        cleanpath = os.path.join(params["clean_proc_dir"], cleanfilename)
        noisepath = os.path.join(params["noise_proc_dir"], noisefilename)

        audio_signals = [noisy_snr, clean_snr, noise_snr]
        file_paths = [noisypath, cleanpath, noisepath]

        file_num += 1
        for i in range(len(audio_signals)):
            try:
                audiowrite(file_paths[i], audio_signals[i], params["fs"])
            except Exception as e:
                print(str(e))

    return (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    )


def main_gen(params):
    """Calls main_gen_single_process() to generate the audio signals, verifies
    that they meet the requirements, and writes the files to storage"""
    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    file_num = params["fileindex_end"] - params["fileindex_start"] + 1

    mp_pool = multiprocessing.Pool(processes=params["n_jobs"])
    mp_args = []

    assert (
        params["fileindex_start"]
        + round(file_num / params["n_jobs"] * params["n_jobs"])
        == params["fileindex_end"] + 1
    )

    for j in range(params["n_jobs"]):
        mp_args.append(
            (
                params,
                j,
                params["fileindex_start"] + round(file_num / params["n_jobs"] * j),
                params["fileindex_start"]
                + round(file_num / params["n_jobs"] * (j + 1)),
                params["n_jobs"],
            )
        )

    mp_results = mp_pool.starmap(main_gen_single_process, mp_args)

    for res in mp_results:
        clean_source_files.extend(res[0])
        clean_clipped_files.extend(res[1])
        clean_low_activity_files.extend(res[2])
        noise_source_files.extend(res[3])
        noise_clipped_files.extend(res[4])
        noise_low_activity_files.extend(res[5])

    return (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    )


def main_body():
    """Main body of this file"""

    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument(
        "--cfg",
        default="noisyspeech_synthesizer.cfg",
        help="Read noisyspeech_synthesizer.cfg for all the details",
    )
    parser.add_argument("--cfg_str", type=str, default="noisy_speech")
    args = parser.parse_args()

    params = dict()
    params["args"] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params["cfg"] = cfg._sections[args.cfg_str]
    cfg = params["cfg"]

    clean_dir = os.path.join(os.path.dirname(__file__), "datasets/clean")

    if cfg["speech_dir"] != "None":
        clean_dir = cfg["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, "Clean speech data is required"

    noise_dir = os.path.join(os.path.dirname(__file__), "datasets/noise")

    if cfg["noise_dir"] != "None":
        noise_dir = cfg["noise_dir"]
    if not os.path.exists:
        assert False, "Noise data is required"

    params["n_jobs"] = int(cfg["nj"])
    params["fs"] = int(cfg["sampling_rate"])
    params["audioformat"] = cfg["audioformat"]
    params["audio_length"] = float(cfg["audio_length"])
    params["silence_length"] = float(cfg["silence_length"])
    params["total_hours"] = float(cfg["total_hours"])

    # clean singing speech
    params["use_singing_data"] = int(cfg["use_singing_data"])
    params["clean_singing"] = str(cfg["clean_singing"])
    params["singing_choice"] = int(cfg["singing_choice"])

    # clean emotional speech
    params["use_emotion_data"] = int(cfg["use_emotion_data"])
    params["clean_emotion"] = str(cfg["clean_emotion"])

    # clean mandarin speech
    params["use_mandarin_data"] = int(cfg["use_mandarin_data"])
    params["clean_mandarin"] = str(cfg["clean_mandarin"])

    # rir
    params["rir_choice"] = int(cfg["rir_choice"])
    params["lower_t60"] = float(cfg["lower_t60"])
    params["upper_t60"] = float(cfg["upper_t60"])
    params["rir_table_csv"] = str(cfg["rir_table_csv"])
    params["clean_speech_t60_csv"] = str(cfg["clean_speech_t60_csv"])

    if cfg["fileindex_start"] != "None" and cfg["fileindex_end"] != "None":
        params["num_files"] = int(cfg["fileindex_end"]) - int(cfg["fileindex_start"])
        params["fileindex_start"] = int(cfg["fileindex_start"])
        params["fileindex_end"] = int(cfg["fileindex_end"])
    else:
        params["num_files"] = int(
            (params["total_hours"] * 60 * 60) / params["audio_length"]
        )
        params["fileindex_start"] = 0
        params["fileindex_end"] = params["num_files"]

    print("Number of files to be synthesized:", params["num_files"])

    params["is_test_set"] = utils.str2bool(cfg["is_test_set"])
    params["clean_activity_threshold"] = float(cfg["clean_activity_threshold"])
    params["noise_activity_threshold"] = float(cfg["noise_activity_threshold"])
    params["snr_lower"] = int(cfg["snr_lower"])
    params["snr_upper"] = int(cfg["snr_upper"])

    params["randomize_snr"] = utils.str2bool(cfg["randomize_snr"])
    params["target_level_lower"] = int(cfg["target_level_lower"])
    params["target_level_upper"] = int(cfg["target_level_upper"])

    if "snr" in cfg.keys():
        params["snr"] = int(cfg["snr"])
    else:
        params["snr"] = int((params["snr_lower"] + params["snr_upper"]) / 2)

    params["noisyspeech_dir"] = utils.get_dir(cfg, "noisy_destination", "noisy")
    params["clean_proc_dir"] = utils.get_dir(cfg, "clean_destination", "clean")
    params["noise_proc_dir"] = utils.get_dir(cfg, "noise_destination", "noise")

    if "speech_csv" in cfg.keys() and cfg["speech_csv"] != "None":
        cleanfilenames = pd.read_csv(cfg["speech_csv"])
        cleanfilenames = cleanfilenames["filename"]
    else:
        # cleanfilenames = glob.glob(os.path.join(clean_dir, params['audioformat']))
        cleanfilenames = []
        for path in Path(clean_dir).rglob("*.wav"):
            cleanfilenames.append(str(path.resolve()))

    shuffle(cleanfilenames)
    #   add singing voice to clean speech
    if params["use_singing_data"] == 1:
        all_singing = []
        for path in Path(params["clean_singing"]).rglob("*.wav"):
            all_singing.append(str(path.resolve()))

        if params["singing_choice"] == 1:  # male speakers
            mysinging = [s for s in all_singing if ("male" in s and "female" not in s)]

        elif params["singing_choice"] == 2:  # female speakers
            mysinging = [s for s in all_singing if "female" in s]

        elif params["singing_choice"] == 3:  # both male and female
            mysinging = all_singing
        else:  # default both male and female
            mysinging = all_singing

        shuffle(mysinging)
        if mysinging is not None:
            all_cleanfiles = cleanfilenames + mysinging
    else:
        all_cleanfiles = cleanfilenames

    #   add emotion data to clean speech
    if params["use_emotion_data"] == 1:
        all_emotion = []
        for path in Path(params["clean_emotion"]).rglob("*.wav"):
            all_emotion.append(str(path.resolve()))

        shuffle(all_emotion)
        if all_emotion is not None:
            all_cleanfiles = all_cleanfiles + all_emotion
    else:
        print("NOT using emotion data for training!")

    #   add mandarin data to clean speech
    if params["use_mandarin_data"] == 1:
        all_mandarin = []
        for path in Path(params["clean_mandarin"]).rglob("*.wav"):
            all_mandarin.append(str(path.resolve()))

        shuffle(all_mandarin)
        if all_mandarin is not None:
            all_cleanfiles = all_cleanfiles + all_mandarin
    else:
        print("NOT using non-english (Mandarin) data for training!")

    params["cleanfilenames"] = all_cleanfiles
    params["num_cleanfiles"] = len(params["cleanfilenames"])
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into
    # subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if "noise_csv" in cfg.keys() and cfg["noise_csv"] != "None":
        noisefilenames = pd.read_csv(cfg["noise_csv"])
        noisefilenames = noisefilenames["filename"]
    else:
        noisefilenames = glob.glob(os.path.join(noise_dir, params["audioformat"]))

    if len(noisefilenames) != 0:
        shuffle(noisefilenames)
        params["noisefilenames"] = noisefilenames
    else:
        noisedirs = glob.glob(os.path.join(noise_dir, "*"))
        if cfg["noise_types_excluded"] != "None":
            dirstoexclude = cfg["noise_types_excluded"].split(",")
            for dirs in dirstoexclude:
                noisedirs.remove(dirs)
        shuffle(noisedirs)
        params["noisedirs"] = noisedirs

    # Call main_gen() to generate audio
    (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    ) = main_gen(params)

    # Create log directory if needed,
    # and write log files of clipped and low activity files
    log_dir = utils.get_dir(cfg, "log_dir", "Logs")

    utils.write_log_file(
        log_dir, "source_files.csv", clean_source_files + noise_source_files
    )
    utils.write_log_file(
        log_dir, "clipped_files.csv", clean_clipped_files + noise_clipped_files
    )
    utils.write_log_file(
        log_dir,
        "low_activity_files.csv",
        clean_low_activity_files + noise_low_activity_files,
    )

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = (
        len(clean_source_files)
        + len(clean_clipped_files)
        + len(clean_low_activity_files)
    )
    total_noise = (
        len(noise_source_files)
        + len(noise_clipped_files)
        + len(noise_low_activity_files)
    )
    pct_clean_clipped = round(len(clean_clipped_files) / total_clean * 100, 1)
    pct_noise_clipped = round(len(noise_clipped_files) / total_noise * 100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files) / total_clean * 100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files) / total_noise * 100, 1)

    print(
        "Of the "
        + str(total_clean)
        + " clean speech files analyzed, "
        + str(pct_clean_clipped)
        + "% had clipping, and "
        + str(pct_clean_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["clean_activity_threshold"] * 100)
        + "% active percentage)"
    )
    print(
        "Of the "
        + str(total_noise)
        + " noise files analyzed, "
        + str(pct_noise_clipped)
        + "% had clipping, and "
        + str(pct_noise_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["noise_activity_threshold"] * 100)
        + "% active percentage)"
    )


if __name__ == "__main__":
    main_body()
