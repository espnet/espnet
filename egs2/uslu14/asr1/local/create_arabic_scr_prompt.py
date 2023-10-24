for split in ["train", "valid"]:
    file = open("dump/dump_arabic_sc/raw/" + split + "/text")
    line_arr = [line for line in file]
    file_write = open("dump/dump_arabic_sc/raw/" + split + "/text", "w")
    for line in line_arr:
        file_write.write(line.split()[0] + " command:" + line.split()[1] + "\n")
    line1_arr = [line.split()[0] + " <|ar|> <|scr|> <|ar_scr|>\n" for line in line_arr]
    file_write = open("dump/dump_arabic_sc/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
