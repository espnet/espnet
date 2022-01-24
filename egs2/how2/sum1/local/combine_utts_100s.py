import os
import sys
import numpy as np
import shutil
import subprocess

utt_data_dir = sys.argv[1]
res_data_dir = utt_data_dir + "_vid100s"
segment_duration = 100


if not os.path.isdir(res_data_dir):
    os.mkdir(res_data_dir)
shutil.copy2(os.path.join(utt_data_dir, "wav.scp"), res_data_dir)


with open(os.path.join(utt_data_dir, "text"), "r") as f:
    utt2text = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }


vid2segments = {}
with open(os.path.join(utt_data_dir, "segments"), "r") as f:
    for line in f.readlines():
        utt_id, rec_id, st_time, end_time = line.strip().split(" ")
        duration = float(end_time) - float(st_time)
        vid_id, index = "_".join(utt_id.split("_")[:-1]), utt_id.split("_")[-1]
        if vid_id in vid2segments:
            vid2segments[vid_id].append(
                [utt_id, float(st_time), float(end_time), utt2text[utt_id], int(index)]
            )
        else:
            vid2segments[vid_id] = [
                [utt_id, float(st_time), float(end_time), utt2text[utt_id], int(index)]
            ]

for vid, segments in vid2segments.items():
    segments.sort(key=lambda x: x[-1])


new_segments = []
new_text = []
new_utt2spk = []
for vid in vid2segments:
    data = vid2segments[vid]

    ## Merge segments of upto "segment_duration" seconds of audio
    while len(data) > 0:
        durations = np.array([x[2] - x[1] for x in data])
        # print("Durations {}".format(durations))
        cumsum = np.cumsum(durations, axis=0)
        # print("CumSum {}".format(cumsum))
        difference = segment_duration - cumsum
        difference[difference < 0] = np.inf
        index = np.argmin(difference)
        # print("Index {}".format(index))
        data_to_combine = data[: index + 1] if index + 1 < len(data) else data
        # print("Data to Combine {}".format(data_to_combine))
        data = data[index + 1 :] if index + 1 < len(data) else []
        combined_start = data_to_combine[0][1]
        combined_end = data_to_combine[index][2]
        combined_text = " ".join([x[3] for x in data_to_combine])
        # print("Combined Start {} | Combined End {} | Combined Duration {} ".format(combined_start,combined_end,combined_end-combined_start))
        new_spkid = vid
        new_uttid = "{}_{:06d}-{:06d}".format(
            new_spkid, int(combined_start * 100), int(combined_end * 100)
        )
        # print("Combining Utts {} into {}".format([x[0] for x in data_to_combine],new_uttid))
        # print("Combined Text {}".format(combined_text))
        new_segments.append(
            "{} {} {:.2f} {:.2f}".format(new_uttid, vid, combined_start, combined_end)
        )
        new_text.append("{} {}".format(new_uttid, combined_text))
        new_utt2spk.append("{} {}".format(new_uttid, new_spkid))

with open(os.path.join(res_data_dir, "segments"), "w") as f:
    f.write("\n".join(new_segments))

with open(os.path.join(res_data_dir, "text"), "w") as f:
    f.write("\n".join(new_text))

with open(os.path.join(res_data_dir, "utt2spk"), "w") as f:
    f.write("\n".join(new_utt2spk))


# spk2utt_command = "utils/utt2spk_to_spk2utt.pl {}/utt2spk > {}/spk2utt".format(utt_data_dir,res_data_dir)
# fix_command = "utils/fix_data_dir.sh {}".format(res_data_dir)

# subprocess.Popen(spk2utt_command)
# subprocess.Popen(fix_command)
