for split in ["train_sp", "dev"]:
    file = open("dump/dump_grabo/raw/" + split + "/text")
    line_arr = [line for line in file]
    line1_arr = [
        line.split()[0] + " <|nl|> <|scr|> <|grabo_scr|>\n" for line in line_arr
    ]
    file_write = open("dump/dump_grabo/raw/" + split + "/prompt", "w")
    for line in line1_arr:
        file_write.write(line)
