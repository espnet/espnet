#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Roshan Sharma (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os 
import sys

data_dir = sys.argv[1]
meeting_audio_dir = sys.argv[2]

out_dir = data_dir + "_summ"


os.makedirs(out_dir, exist_ok=True)
os.makedirs(meeting_audio_dir, exist_ok=True)

with open(os.path.join(data_dir, "wav.scp"), "r") as f:
    rec2wav = {
        line.strip().split(' ')[0]:line.strip().split(' ')[8]
          for line in f.readlines()
          }

with open(os.path.join(data_dir, "text"), "r") as f:
    utt2text = {
        line.strip().split(' ')[0]:" ".join(line.strip().split(' ')[1:])
          for line in f.readlines()
          }


with open(os.path.join(data_dir, "segments"), "r") as f:
    utt2segs = {
        line.strip().split(' ')[0]:line.strip().split(" ") + [utt2text[line.strip().split(' ')[0]]]
          for line in f.readlines()
          }

meeting2segs = {}
for utt, segs in utt2segs.items():
    meeting_id = utt.split('_')[1]
    if meeting_id not in meeting2segs:
        meeting2segs[meeting_id] = []
    meeting2segs[meeting_id].append(segs)



wav_scp = []
segments = []
utt2spk = []
text = []
cmd = []



utt_dir = os.path.join(os.path.dirname(meeting_audio_dir),"audio_utts")

with open(os.path.join(out_dir, "wav.scp"), "a") as f,open("cmd", "a") as g, open(os.path.join(out_dir, "segments"), "a") as f1, open(os.path.join(out_dir, "utt2spk"), "a") as f2, open(os.path.join(out_dir, "text"), "a") as f3:
    for meeting,segs in meeting2segs.items():
        segs.sort(key=lambda x: float(x[2]))
        transcript = " ".join([seg[-1] for seg in segs])
        output_audio_path = os.path.join(meeting_audio_dir, meeting + ".wav")
        
        with open(os.path.join(meeting_audio_dir, f"{meeting}.list"), "w") as f4:
            f4.write("\n".join([f"file '{utt_dir}/{seg[0]}.wav'" for seg in segs]))

        f.write(f"{meeting} {output_audio_path}\n")
        f1.write(f"{meeting} {meeting} 0 0\n")
        f2.write(f"{meeting} {meeting}\n")
        f3.write(f"{meeting} {transcript}\n")
        f2.write("\n".join(utt2spk))
        f3.write("\n".join(text))
        g.write("ffmpeg -y -safe 0 -hide_banner -loglevel error -f concat -i {} -c copy {}\n".format(os.path.join(meeting_audio_dir, f"{meeting}.list"), output_audio_path))
