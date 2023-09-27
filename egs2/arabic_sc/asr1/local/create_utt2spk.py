for split in ["train", "valid", "test"]:
    line_arr = [line for line in open("data/" + split + "/wav.scp")]
    file_write = open("data/" + split + "/utt2spk", "w")
    line1_arr = []
    for line in line_arr:
        file_write.write(line.split()[0] + " " + line.split()[0] + "\n")
