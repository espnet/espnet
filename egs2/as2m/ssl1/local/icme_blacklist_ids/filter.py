# filter_feats.py

# Paths = ALWAYS ABSOLUTE
blacklist_file = "/work/nvme/bbjs/sbharadwaj/espnet/egs2/as2m/ssl1/local/icme_blacklist_ids/youtube_ids.txt"  # your blacklist ids file
feats_file = "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/fbank/yt8m/feats.scp"  # your feats.scp file
output_file = "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/fbank/7Msounds/yt8m.filtered.scp"  # output filtered feats.scp file

# Step 1: Read blacklist ids into a set
with open(blacklist_file, "r") as f:
    blacklist_ids = set(line.strip() for line in f if line.strip())

# Step 2: Open feats.scp, filter, and write output
with open(feats_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        utt_id = line.strip().split()[0]  # take the first column (utterance ID)
        if "yodas" or "vggsound" in feats_file:
            youtube_id = utt_id[:11].strip()
        elif "yt8m" in feats_file:
            youtube_id = utt_id[-11:].strip()

        assert len(youtube_id) == 11, "Youtube ID should be 11 characters long"
        if youtube_id not in blacklist_ids:
            fout.write(line)

print(f"Filtered feats.scp written to {output_file}")
