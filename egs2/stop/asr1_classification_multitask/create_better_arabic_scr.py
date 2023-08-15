# replace_dict={}
# replace_dict["<neu>"]="em:neu SEP"
# replace_dict["<ang>"]="em:ang SEP"
# replace_dict["<sad>"]="em:sad SEP"
# replace_dict["<hap>"]="em:hap SEP"
# for split in ["train","valid"]:
#     file=open("dump_iemocap/dump/raw/"+split+"/text")
#     line_arr=[line for line in file]
#     line1_arr=[]
#     for line in line_arr:
#         for k in replace_dict:
#             if k in line:
#                 line1=line.replace(k,replace_dict[k])
#         # print(line1.split("em:"))
#         assert len(line1.split("em:"))==2
#         line1_arr.append(line1)
#     file_write=open("dump_iemocap/dump/raw/"+split+"/text_new","w")
#     for line in line1_arr:
#         file_write.write(line)

for split in ["train","valid"]:
    file=open("dump_arabic_sc/dump/raw/"+split+"/text")
    line_arr=[line for line in file]
    file_write=open("dump_arabic_sc/dump/raw/"+split+"/text_new","w")
    for line in line_arr:
        file_write.write(line.split()[0]+" command:"+line.split()[1]+"\n")
    line1_arr=[line.split()[0]+" <|ar|> <|scr|> <|ar_scr|>\n" for line in line_arr]
    file_write=open("dump_arabic_sc/dump/raw/"+split+"/prompt","w")
    for line in line1_arr:
        file_write.write(line)