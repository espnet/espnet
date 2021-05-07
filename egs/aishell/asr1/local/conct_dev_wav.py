import sys
import pdb

before = 0
after = 0

utts = {}
# sox track1.wav silence.wav track2.wav silence.wav ... output.wav
wav_file = open(sys.argv[1], "r")
orig_wavs = wav_file.readlines()

sdir = sys.argv[2]
sfile = sys.argv[3]
wfile = sys.argv[4]

for line in orig_wavs:
    before += 1
    splited = line.strip().split()
    uid, utt = splited[0], splited[1:]
    sid = uid.split("W")[0]
    if sid in utts.keys():
        utts[sid].append(utt[0])
    else:
        utts[sid] = utt

utts_2 = {}
for sid in utts.keys():
    utts_2[sid + "A"] = utts[sid][len(utts[sid]) // 2 :]
    utts_2[sid + "B"] = utts[sid][: len(utts[sid]) // 2]


f = open(sfile, "a")
for sid in utts_2.keys():
    cmd = "sox " + " ".join(utts_2[sid]) + (" -t wav " + sdir + "/" + sid + ".wav")
    f.write(cmd + "\n")
f.close()

f = open(wfile, "a")
for sid in utts_2.keys():
    cmd = sid + " " + sdir + "/" + sid + ".wav"
    f.write(cmd + "\n")
f.close()
