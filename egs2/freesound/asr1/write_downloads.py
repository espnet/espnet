for count in range(1,33):
    file=open("dump/raw/org/train/logs/wav."+str(count)+".scp")
    line_arr=[line for line in file]
    file_write=open("write_downloads"+str(count)+".sh","w")
    for line in line_arr:
        if len(" ".join(line.split()[1:]).split('"'))>3:
            # import pdb;pdb.set_trace()
            a1=('\\"').join(" ".join(line.split()[1:]).split('"')[1:-1])
            line1=a1[2:].split("trim")[0]+'"downloads/'+line.split()[0]+'.wav" trim'+" ".join(line.split()[1:]).split('"')[-2][2:].split("trim")[-1]
        else:
            line1=" ".join(line.split()[1:]).split('"')[-2][2:].split("trim")[0]+'"downloads/'+line.split()[0]+'.wav" trim'+" ".join(line.split()[1:]).split('"')[-2][2:].split("trim")[-1]
        line2='sox "/'+"/".join(line1.split("/")[1:]).split(".wav")[0]+'.wav"'+".wav".join("/".join(line1.split("/")[1:]).split(".wav")[1:])
        file_write.write(line2+"\n")