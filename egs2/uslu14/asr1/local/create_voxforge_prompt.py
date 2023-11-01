accent_dict = {}
for split in ["train", "valid"]:
    file = open("dump/dump_voxforge/raw/" + split + "/text")
    line_arr = [line for line in file]
    file_write = open("dump/dump_voxforge/raw/" + split + "/text", "w")
    line1_arr = []
    for line in line_arr:
        file_write.write(line.split()[0] + " lang:" + line.split()[1] + "\n")
        line1_arr.append(
            line.split()[0] + " <|" + line.split()[1] + "|> <|lid|> <|voxforge|>\n"
        )
        if "lang:" + line.split()[1] not in accent_dict:
            accent_dict["lang:" + line.split()[1]] = 1
    file_write = open("dump/dump_voxforge/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
for k in accent_dict:
    print(k)
