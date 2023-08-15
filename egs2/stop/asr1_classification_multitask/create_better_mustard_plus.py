accent_dict={}
for split in ["train","valid"]:
    file=open("dump_mustard_plus_plus/dump/raw/"+split+"/text")
    line_arr=[line for line in file]
    file_write=open("dump_mustard_plus_plus/dump/raw/"+split+"/text_new","w")
    for line in line_arr:
        file_write.write(line)
        if line.split()[1] not in accent_dict:
            accent_dict[line.split()[1]]=1
    line1_arr=[line.split()[0]+" <|en|> <|scd|> <|mustard_plus_plus|>\n" for line in line_arr]
    file_write=open("dump_mustard_plus_plus/dump/raw/"+split+"/prompt","w")
    for line in line1_arr:
        file_write.write(line)
for k in accent_dict:
    print(k)