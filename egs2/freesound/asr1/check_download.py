import os
list1=os.listdir('downloads/')
dict1={}
for k in list1:
    dict1[k]=1
print(len(list1))
print(len(dict1))
# exit()
dict2={}
for count in range(1,33):
    file=open("dump/raw/org/train/logs/wav."+str(count)+".scp")
    line_arr=[line for line in file]
    print(len(line_arr))
    # file_write=open("write_downloads"+str(count)+".sh","w")
    for line in line_arr:
        if line.split()[0]+'.wav' not in dict1:
            print(line.split()[0]+'.wav')
            # exit()
        if line.split()[0]+'.wav' not in dict2:
            dict2[line.split()[0]+'.wav']=1
        else:
            print(line.split()[0]+'.wav')
            print("duplicate")
for k in dict1:
    if k not in dict2:
        print(k)
print(len(dict2))
