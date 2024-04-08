intent_dict = {}
for split in ["train", "valid"]:
    file = open("dump/dump_fsc/raw/" + split + "/text")
    line_arr = [line for line in file]
    line1_arr = []
    for line in line_arr:
        if line.split()[1] not in intent_dict:
            intent_dict[line.split()[1]] = 1
        line1 = (
            line.split()[0]
            + " in:"
            + line.split()[1]
            + " SEP "
            + " ".join(line.split()[2:]).lower()
            + "\n"
        )
        line1_arr.append(line1)
    file_write = open("dump/dump_fsc/raw/" + split + "/text", "w")
    for line in line1_arr:
        file_write.write(line)
    print(intent_dict)
    line1_arr = [
        line.split()[0] + " <|en|> <|ic|> <|grabo_scr|>\n" for line in line_arr
    ]

for split in ["train", "valid"]:
    file = open("dump/dump_fsc/raw/" + split + "/text")
    line_arr = [line for line in file]
    line1_arr = [line.split()[0] + " <|en|> <|ic|> <|fsc|>\n" for line in line_arr]
    file_write = open("dump/dump_fsc/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
