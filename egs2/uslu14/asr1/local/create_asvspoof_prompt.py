accent_dict = {}
for split in ["train", "valid"]:
    file = open("dump/dump_asvspoof/raw/" + split + "/text")
    line_arr = [line for line in file]
    file_write = open("dump/dump_asvspoof/raw/" + split + "/text", "w")
    for line in line_arr:
        file_write.write(line.split()[0] + " class:" + line.split()[1] + "\n")
        if "class:" + line.split()[1] not in accent_dict:
            accent_dict["class:" + line.split()[1]] = 1
    line1_arr = [
        line.split()[0] + " <|en|> <|fsd|> <|asvspoof|>\n" for line in line_arr
    ]
    file_write = open("dump/dump_asvspoof/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
for k in accent_dict:
    print(k)
