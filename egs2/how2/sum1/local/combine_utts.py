import sys
import os
import fileinput

vid2utts = {}
for line in fileinput.input():
    vid_id,index = "_".join(line.strip().split(" ")[0].split("_")[:-1]), line.strip().split(" ")[0].split("_")[-1]
    text = " ".join(line.strip().split(" ")[1:])
    index=int(index)
    if vid_id in vid2utts:
        vid2utts[vid_id].append((index,text))
    else:
        vid2utts[vid_id] = [(index,text)]
    
for vid,utts in vid2utts.items():
    utts.sort(key=lambda x : x[0])
    text = " ".join([x[1] for x in utts])
    print("{} {}".format(vid,text))
