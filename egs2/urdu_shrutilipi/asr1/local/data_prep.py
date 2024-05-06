f_init = open("shrutilipi_fairseq/urdu/train.wrd").readlines()
f1_init = open("shrutilipi_fairseq/urdu/train.tsv").readlines()

import os

dict_data = {}
for a, b in zip(f_init, f1_init[1:]):
    dict_data[b] = a
sorted_dict = dict(sorted(dict_data.items()))

f1 = list(sorted_dict.keys())
f = list(sorted_dict.values())
train_len = 78551
valid_len = 88370
test_len = 98189
i = 1
o = open("data/train/text", "w")
o1 = open("data/train/wav.scp", "w")
o2 = open("data/train/utt2spk", "w")
o3 = open("data/train/spk2utt", "w")

text_set = set()

for l, l1 in zip(f[:78551], f1[:78551]):
    path, frame_num = l1.split("\t")
    if l in text_set:
        continue
    if not os.path.exists("newsonair_v5/urdu/" + path):
        continue
    text_set.add(l)
    o.write(path + " " + l)
    o1.write(path + " " + "newsonair_v5/urdu/" + path + "\n")
    o2.write(path + " " + path + "\n")
    o3.write(path + " " + path + "\n")
    i = i + 1

o.close()
o1.close()
o2.close()
o3.close()

o = open("data/dev/text", "w")
o1 = open("data/dev/wav.scp", "w")
o2 = open("data/dev/utt2spk", "w")
o3 = open("data/dev/spk2utt", "w")

text_set = set()

for l, l1 in zip(f[train_len:valid_len], f1[train_len:valid_len]):
    path, frame_num = l1.split("\t")
    if l in text_set:
        continue
    if not os.path.exists("newsonair_v5/urdu/" + path):
        continue
    text_set.add(l)
    o.write(path + " " + l)
    o1.write(path + " " + "newsonair_v5/urdu/" + path + "\n")
    o2.write(path + " " + path + "\n")
    o3.write(path + " " + path + "\n")
    i = i + 1

o.close()
o1.close()
o2.close()
o3.close()

o = open("data/test/text", "w")
o1 = open("data/test/wav.scp", "w")
o2 = open("data/test/utt2spk", "w")
o3 = open("data/test/spk2utt", "w")


text_set = set()

for l, l1 in zip(f[valid_len:test_len], f1[valid_len:test_len]):
    path, frame_num = l1.split("\t")
    if not os.path.exists("newsonair_v5/urdu/" + path):
        continue
    if l in text_set:
        continue
    text_set.add(l)
    o.write(path + " " + l)
    o1.write(path + " " + "ewsonair_v5/urdu/" + path + "\n")
    o2.write(path + " " + path + "\n")
    o3.write(path + " " + path + "\n")
    i = i + 1

o.close()
o1.close()
o2.close()
o3.close()
