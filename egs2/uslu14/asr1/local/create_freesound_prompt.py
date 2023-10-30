accent_dict = {}
for split in ["train", "valid"]:
    file = open("dump/dump_freesound/raw/" + split + "/text")
    line_arr = [line for line in file]
    file_write = open("dump/dump_freesound/raw/" + split + "/text", "w")
    line1_arr = []
    for line in line_arr:
        file_write.write(line)
        if line.split()[1] not in accent_dict:
            accent_dict[line.split()[1]] = 1
        if line.split()[1] == "vad_class:speech":
            line1_arr.append(line.split()[0] + " <|en|> <|vad|> <|freesound|>\n")
        else:
            line1_arr.append(line.split()[0] + " <|nospeech|>\n")
    file_write = open("dump/dump_freesound/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
for k in accent_dict:
    print(k)
