for split in ["train_sp", "dev"]:
    file = open("dump/dump_speechcommands/raw/" + split + "/text")
    line_arr = [line for line in file]
    file_write = open("dump/dump_speechcommands/raw/" + split + "/text", "w")
    for line in line_arr:
        file_write.write(line.split()[0] + " command:" + line.split()[1] + "\n")
    line1_arr = [
        line.split()[0] + " <|en|> <|scr|> <|google_scr|>\n" for line in line_arr
    ]
    file_write = open("dump/dump_speechcommands/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
