file=open("../asr1/dump/raw/valid/text")
line_arr=[line for line in file]
file_write=open("dump/raw/valid/text_stop","w")
for line in line_arr:
    line1=[]
    for k in line.split():
        if k[:4]=="[in:" or k[:4]=="[sl:":
            line1.append(k+"_SLURP")
        else:
            line1.append(k)
    # import pdb;pdb.set_trace()
    file_write.write(" ".join(line1)+'\n')

# for k in intent_dict:
#     print(intent_dict[k])
# for k in slot_dict:
#     print(slot_dict[k])