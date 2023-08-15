token_file=open("data/en_token_list/whisper_multilingual/tokens.txt")
token_list=[line.strip() for line in token_file]
file=open("exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_vad_asr_model_valid.acc.ave/test_freesound/token_int")
line_arr=[line for line in file]
result_dict={}
import pdb;pdb.set_trace()
for line in line_arr:    
    if token_list[int(line.split()[1])]=="<|nospeech|>":
        result_dict[line.split()[0]]=1
    else:
        result_dict[line.split()[0]]=0
gt_file=open("dump/raw/test_freesound/text")
gt_line_arr=[line for line in gt_file]
gt_dict={}
import pdb;pdb.set_trace()
for line in gt_line_arr:    
    if line.split()[1]=="vad_class:background":
        gt_dict[line.split()[0]]=1
    else:
        gt_dict[line.split()[0]]=0
error=0
for k in result_dict:
    if gt_dict[k]!=result_dict[k]:
        error+=1
print(error/len(result_dict))

# line_arr=[line.replace("<|er|>","").replace(" SEP","") for line in line_arr]
# line1_arr=[]
# for line in line_arr:
#     line1=line
#     for k in replace_dict:
#         if k in line:
#             line1=line.replace(k,replace_dict[k])
#     line1_arr.append(line1)
# file_write=open("exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_er_asr_model_valid.acc.ave/test_iemocap/text_new","w")
# for line in line1_arr:
#     file_write.write(line)
