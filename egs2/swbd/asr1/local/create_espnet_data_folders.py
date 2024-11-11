# Copyright 2024  Siddhant Arora
#           2024  Carnegie Mellon University
# Apache 2.0

import os
import re
import sys

import pandas as pd

dir_dict = {
    "train": "Train_Two_Channel_Label_Mono_subsample_update.csv",
    "valid": "Val_Two_Channel_Label_Mono_subsample_update.csv",
    "test": "Val_Two_Channel_Label_Mono_update.csv",
}

for x in dir_dict:
    line_count=0
    word_dict={}
    for line in open("data/"+dir_dict[x]):
        line1=line.strip().split(",")
        if (("sw0"+line1[0]) not in word_dict):
            word_dict["sw0"+line1[0]]={}
        if float(line1[1]) not in word_dict["sw0"+line1[0]]:
            word_dict["sw0"+line1[0]][float(line1[1])]={}
        if float(line1[2]) not in word_dict["sw0"+line1[0]][float(line1[1])]:
            word_dict["sw0"+line1[0]][float(line1[1])][float(line1[2])]=line1
        else:
            print("error")
            import pdb;pdb.set_trace()

    for k in word_dict:
        for j in word_dict[k]:
            word_dict[k][j]=dict(sorted(word_dict[k][j].items()))
        word_dict[k]=dict(sorted(word_dict[k].items()))

    file_write=open("data/"+x+"/wav.scp","w")
    id1_dict={}
    for line in open("data/train_nodup/wav.scp"):
        id1=line.split("-")[0]
        line1=" ".join(line.split()[1:]).replace("-c 1 ","").replace("-c 2 ","").replace("rate 16000","rate 16000 channels 1")
        if id1 in word_dict:
            if id1 not in id1_dict:
                id1_dict[id1]=1
                file_write.write(id1+" "+line1+"\n")

    file_write=open("data/"+x+"/wav.scp.bak","w")
    id1_dict={}
    for line in open("data/train_nodup/wav.scp.bak"):
        id1=line.split("-")[0]
        line1=" ".join(line.split()[1:]).replace("-c 1 ","").replace("-c 2 ","")
        if id1 in word_dict:
            if id1 not in id1_dict:
                id1_dict[id1]=1
                file_write.write(id1+" "+line1+"\n")

    if x=="test":
        file_text_write=open("data/"+x+"/text","w")
        file_spk_write=open("data/"+x+"/utt2spk","w")
        for id1 in id1_dict:
            file_spk_write.write(id1+" "+id1+"\n")
            seg_text_str=""
            for j in word_dict[id1]:
                for j_end in word_dict[id1][j]:
                    k=word_dict[id1][j][j_end]
                    start_time=k[1]
                    end_time=k[2]
                    if ((float(end_time))<30):
                        start_time_write="{:.2f}".format(0.0)
                    else:
                        start_time_write="{:.2f}".format(float(end_time)-30)
                    end_time_write="{:.2f}".format(float(end_time))
                    seg_text_str+=" "+k[-2]
            file_text_write.write(id1+seg_text_str+"\n")               
    else:
        file_write=open("data/"+x+"/segments","w")
        file_text_write=open("data/"+x+"/text","w")
        file_spk_write=open("data/"+x+"/utt2spk","w")
        for id1 in id1_dict:
            for j in word_dict[id1]:
                for j_end in word_dict[id1][j]:
                    k=word_dict[id1][j][j_end]
                    start_time=k[1]
                    end_time=k[2]
                    if ((float(end_time))<30):
                        start_time_write="{:.2f}".format(0.0)
                    else:
                        start_time_write="{:.2f}".format(float(end_time)-30)
                    end_time_write="{:.2f}".format(float(end_time))
                    segment_id=id1+"_"+"{:.6f}".format(float(j)/10**4).split(".")[-1]+"_"+"{:.6f}".format(float(end_time)/10**4).split(".")[-1]
                    file_write.write(segment_id+" "+id1+" "+start_time_write+" "+end_time_write+"\n")
                    file_text_write.write(segment_id+" "+k[-2]+"\n")
                    file_spk_write.write(segment_id+" "+id1+"\n")
