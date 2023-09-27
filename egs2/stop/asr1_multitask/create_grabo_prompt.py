for split in ["train_sp", "dev"]:
    file = open("dump_grabo/dump/raw/" + split + "/text")
    line_arr = [line for line in file]
    line1_arr = [
        line.split()[0] + " <|nl|> <|scr|> <|grabo_scr|>\n" for line in line_arr
    ]
    file_write = open("dump_grabo/dump/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
