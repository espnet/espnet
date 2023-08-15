for split in ["train_combined","valid"]:
    file1=open("dump/raw/"+split+"/prompt")
    file1_new=open("dump/raw/"+split+"/prompt_new","w")
    for line in file1:
        line_id=line.split()[0]
        prompt=" ".join(line.split()[1:])
        if prompt=="<|en|> <|scr|>":
            prompt+=" <|google_scr|>"
        elif prompt=="<|nl|> <|scr|>":
            prompt+=" <|grabo_scr|>"
        elif prompt=="<|lt|> <|scr|>":
            prompt+=" <|lt_scr|>"
        elif prompt=="<|ar|> <|scr|>":
            prompt+=" <|ar_scr|>"
        elif prompt=="<|en|> <|ic|>":
            prompt+=" <|fsc|>"
        elif "<|lid|>" in prompt:
            prompt+=" <|voxforge|>"
        elif prompt=="<|en|> <|fsd|>":
            prompt+=" <|asvspoof|>"
        elif prompt=="<|en|> <|er|>":
            prompt+=" <|iemocap|>"
        elif prompt=="<|en|> <|accent_rec|>":
            prompt+=" <|accentdb|>"
        elif prompt=="<|en|> <|scd|>":
            prompt+=" <|mustard|>"
        file1_new.write(line_id+" "+prompt+"\n")