dir_dict={}
dir_dict["train"]="downloads/"
dir_dict["valid"]="valid_downloads/"
dir_dict["test"]="test_downloads/"
for split in ["train","valid","test"]:
    file=open("data/"+split+"/wav.scp")
    line_arr=[line for line in file]
    file_write=open("data/"+split+"/wav.scp_new","w")
    for line in line_arr:
        # print(line)
        # import pdb;pdb.set_trace()
        line1=line.split()[0]+' '+dir_dict[split]+line.split()[0]+'.wav'
        file_write.write(line1+"\n")