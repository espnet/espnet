for split in ["train_sp", "dev"]:
    file = open("dump_speechcommands/dump/raw/" + split + "/text")
    line_arr = [line for line in file]
    file_write = open("dump_speechcommands/dump/raw/" + split + "/text", "w")
    for line in line_arr:
        file_write.write(line.split()[0] + " command:" + line.split()[1] + "\n")
    line1_arr = [
        line.split()[0] + " <|en|> <|scr|> <|google_scr|>\n" for line in line_arr
    ]
    file_write = open("dump_speechcommands/dump/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
