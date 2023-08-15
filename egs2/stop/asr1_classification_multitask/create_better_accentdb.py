accent_dict={}
for split in ["train","valid"]:
    file=open("dump_accentdb/dump/raw/"+split+"/text")
    line_arr=[line for line in file]
    file_write=open("dump_accentdb/dump/raw/"+split+"/text_new","w")
    for line in line_arr:
        file_write.write(line.split()[0]+" accent:"+line.split()[1]+"\n")
        if "accent:"+line.split()[1] not in accent_dict:
            accent_dict["accent:"+line.split()[1]]=1
    line1_arr=[line.split()[0]+" <|en|> <|accent_rec|> <|accentdb|>\n" for line in line_arr]
    file_write=open("dump_accentdb/dump/raw/"+split+"/prompt","w")
    for line in line1_arr:
        file_write.write(line)
for k in accent_dict:
    print(k)