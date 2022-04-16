# This script removes very few examples (less than 0.1%) of the train and devtest sets
# Those few examples contained corrupted utterance Id or empty transcripts.


import os

base = "downloads/African_Accented_French/"
existing = []
for _, _, f in os.walk(base + "speech/train/ca16"):
    for fi in f:
        existing.append(fi[:-4])

old_f = open(base + "transcripts/train/ca16_conv/transcripts.txt")
new_f = open(
    base + "transcripts/train/ca16_conv/new_transcripts.txt",
    "w",
)

for row in old_f:
    if row.split(" ")[0][:-4] in existing:
        new_f.write(row)

old_f = open(base + "transcripts/train/ca16_read/conditioned.txt")
new_f = open(
    base + "transcripts/train/ca16_read/new_conditioned.txt",
    "w",
)

for row in old_f:
    if row.split(" ")[0] in existing:
        new_f.write(row)

existing = []
for _, _, f in os.walk(base + "speech/devtest/ca16"):
    for fi in f:
        existing.append(fi[:-4])

old_f = open(base + "transcripts/devtest/ca16_read/conditioned.txt")
new_f = open(
    base + "transcripts/devtest/ca16_read/new_conditioned.txt",
    "w",
)

for row in old_f:
    if row.split(" ")[0] in existing:
        new_f.write(row)
