for split in ["train", "valid"]:
    file = open("dump/dump_lt_speech_commands/raw/" + split + "/text")
    line_arr = [line for line in file]
    line1_arr = [line.split()[0] + " <|lt|> <|scr|> <|lt_scr|>\n" for line in line_arr]
    file_write = open("dump/dump_lt_speech_commands/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
