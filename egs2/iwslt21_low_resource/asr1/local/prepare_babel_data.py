#!/usr/bin/env python3

# Copyright 2021 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import glob
import os
import sys

idir = sys.argv[1]


def read_transcription(txt):
    utts = []
    text = ""
    prev_sec = 0.0
    started = False

    with open(txt) as f:
        for line in f:
            if line[0] == "[":
                sec = float(line[1:-2])
                if started:
                    text = text.lower()
                    text = text.replace("- ", " ")
                    text = text.replace("*", " ")
                    text = text.replace("(())", " ")
                    text = text.replace("~", " ")
                    text = text.replace("_", " ")
                    text = text.replace("á", "a")
                    text = text.replace("é", "e")
                    words = [
                        w if w[0] == "<" else w.replace("-", " ") for w in text.split()
                    ]
                    words = " ".join(words).split()
                    if len(words) > 0 and words[0] != "<no-speech>":
                        utts.append(
                            {"start": prev_sec, "text": " ".join(words), "end": sec}
                        )
                    started = False
                else:
                    prev_sec = sec
                    text = ""
                    started = True
            else:
                text += " " + line

    return utts


mysubsets = {"training": "train", "dev": "valid"}

for subset in mysubsets.keys():
    odir = "data/{}_babel".format(mysubsets[subset])
    os.makedirs(odir, exist_ok=True)

    with open(odir + "/text", "w", encoding="utf-8") as text, open(
        odir + "/wav.scp", "w"
    ) as wavscp, open(odir + "/utt2spk", "w") as utt2spk, open(
        odir + "/segments", "w"
    ) as segments:
        for part in ["scripted", "conversational"]:
            for audio in glob.glob(os.path.join(idir, part, subset, "audio", "*.sph")):
                recoid = os.path.split(audio)[1][:-4]
                wavscp.write(
                    f"{recoid} ffmpeg -i {audio}"
                    + " -acodec pcm_s16le -ar 16000 -ac 1 -f wav - |\n"
                )
                transcription = os.path.join(
                    idir, part, subset, "transcription", recoid + ".txt"
                )
                utts = read_transcription(transcription)
                for utt in utts:
                    uttid = "{}_{:06d}_{:06d}".format(
                        recoid, int(utt["start"] * 100), int(utt["end"] * 100)
                    )
                    text.write("{} {}\n".format(uttid, utt["text"]))
                    utt2spk.write("{} {}\n".format(uttid, uttid))
                    segments.write(
                        "{} {} {} {}\n".format(uttid, recoid, utt["start"], utt["end"])
                    )
