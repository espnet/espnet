#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Note (jiatong): Credit to https://github.com/hitachi-speech/EEND
#
# This script generates simulated multi-talker mixtures for diarization
# (No speaker overlaps)
#
# common/make_mixture_nooverlap.py \
#     mixture.scp \
#     data/mixture \
#     wav/mixture


import argparse
import json
import math
import os

import common
import numpy as np
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("script", help="list of json")
parser.add_argument("out_data_dir", help="output data dir of mixture")
parser.add_argument("out_wav_dir", help="output mixture wav files are stored here")
parser.add_argument("--rate", type=int, default=16000, help="sampling rate")
args = parser.parse_args()

# open output data files
segments_f = open(args.out_data_dir + "/segments", "w")
utt2spk_f = open(args.out_data_dir + "/utt2spk", "w")
wav_scp_f = open(args.out_data_dir + "/wav.scp", "w")

# outputs are resampled at target sample rate
resample_cmd = "sox -t wav - -t wav - rate {}".format(args.rate)

for line in open(args.script):
    recid, jsonstr = line.strip().split(None, 1)
    indata = json.loads(jsonstr)
    recid = indata["recid"]
    noise = indata["noise"]
    noise_snr = indata["snr"]
    mixture = []
    data = []
    pos = 0
    for utt in indata["utts"]:
        spkid = utt["spkid"]
        wav = utt["utt"]
        interval = utt["interval"]
        rir = utt["rir"]
        st = 0
        et = None
        if "st" in utt:
            st = np.rint(utt["st"] * args.rate).astype(int)
        if "et" in utt:
            et = np.rint(utt["et"] * args.rate).astype(int)
        silence = np.zeros(int(interval * args.rate))
        data.append(silence)
        # utterance is reverberated using room impulse response
        if rir:
            preprocess = (
                "wav-reverberate --print-args=false "
                " --impulse-response={} - -".format(rir)
            )
            wav_rxfilename = common.process_wav(wav, preprocess)
        else:
            wav_rxfilename = wav
        wav_rxfilename = common.process_wav(wav_rxfilename, resample_cmd)
        speech, _ = common.load_wav(wav_rxfilename, st, et)
        data.append(speech)
        # calculate start/end position in samples
        startpos = pos + len(silence)
        endpos = startpos + len(speech)
        # write segments and utt2spk
        uttid = "{}_{}_{:07d}_{:07d}".format(
            spkid, recid, int(startpos / args.rate * 100), int(endpos / args.rate * 100)
        )
        print(uttid, recid, startpos / args.rate, endpos / args.rate, file=segments_f)
        print(uttid, spkid, file=utt2spk_f)
        pos = endpos
    mixture = np.concatenate(data)
    maxlen = len(mixture)
    # noise is repeated or cutted for fitting to the mixture data length
    noise_resampled = common.process_wav(noise, resample_cmd)
    noise_data, _ = common.load_wav(noise_resampled)
    if maxlen > len(noise_data):
        noise_data = np.pad(noise_data, (0, maxlen - len(noise_data)), "wrap")
    else:
        noise_data = noise_data[:maxlen]
    # noise power is scaled according to selected SNR, then mixed
    signal_power = np.sum(mixture**2) / len(mixture)
    noise_power = np.sum(noise_data**2) / len(noise_data)
    scale = math.sqrt(math.pow(10, -noise_snr / 10) * signal_power / noise_power)
    mixture += noise_data * scale
    # output the wav file and write wav.scp
    outfname = "{}.wav".format(recid)
    outpath = os.path.join(args.out_wav_dir, outfname)
    sf.write(outpath, mixture, args.rate)
    print(recid, os.path.abspath(outpath), file=wav_scp_f)

wav_scp_f.close()
segments_f.close()
utt2spk_f.close()
