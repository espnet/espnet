import json
import sys
import argparse
import os

# SUBWORD MAJORITY VOTE
def generate_entity_file(line_arr, asr_line_arr, output_file="result_test.json"):
    fp = open(output_file, "w")
    for line_count in range(len(line_arr)):
        ent_final_arr = []
        ner_line=line_arr[line_count].strip().split(" ")[1:]
        asr_line=asr_line_arr[line_count].strip().split(" ")[1:]
        if "<sos/eos>" in asr_line[0]:
            asr_line=asr_line[1:-1]
            ner_line=ner_line[:-1]
        # if "<sos/eos>" in asr_line:
        #     print(asr_line)
        #     exit()
        # print(asr_line)
        # print(ner_line)
        if len(ner_line)==(len(asr_line)+1):
            ner_line=ner_line[:-1]
        if len(asr_line)!=len(ner_line):
            print(asr_line)
            print(ner_line)
            print(len(asr_line))
            print(len(ner_line))
            print(line_arr[line_count].strip().split(" ")[0])
        else:
            word_count=0
            while True:
                if word_count>=len(ner_line):
                    break
                ner_label=ner_line[word_count]
                if ner_label=="na":
                    word_count+=1
                    continue
                if ner_label=="FILL" and asr_line[word_count][0]!="▁":
                    word_count+=1
                    continue
                if asr_line[word_count][0]!="▁":
                    print("weird")
                    # print(asr_line)
                    # print(asr_line[word_count])
                    word_count+=1
                    continue
                if ner_label=="FILL":
                    print("FILL should not have been here")
                    word_count+=1
                    continue
                if "_I" in ner_label:
                    word_count+=1
                    continue
                ent_type = ner_label.replace("_B","")
                ent_val = asr_line[word_count][1:]
                while True:
                    word_count+=1
                    if word_count>=len(ner_line):
                        break
                    if word_count>=len(asr_line):
                        break
                    if asr_line[word_count][0]=="▁":
                        if "_I" not in ner_line[word_count]:
                            break   
                        ent_val+= " " + asr_line[word_count][1:]             
                    else:
                        ent_val+=asr_line[word_count]
                dict1 = {}
                dict1["type"] = ent_type
                dict1["filler"] = ent_val
                ent_final_arr.append(dict1)
        file_name = line_arr[line_count].strip().split(" ")[0].split("_")[-1].replace(")", "")
        file_name = "audio-" + file_name + ".flac"
        write_dict = {}
        write_dict["text"] = ""
        write_dict["scenario"] = ""
        write_dict["action"] = ""
        write_dict["entities"] = ent_final_arr
        write_dict["file"] = file_name
        json.dump(write_dict, fp)
        fp.write("\n")


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
parser.add_argument(
    "--test_folder",
    default="inference_st_model_valid.acc.ave/test/",
    help="Directory inside exp_root containing inference on test set",
)
parser.add_argument(
    "--valid_folder",
    default="inference_st_model_valid.acc.ave/devel/",
    help="Directory inside exp_root containing inference on valid set",
)

args = parser.parse_args()

exp_root = args.exp_root
valid_inference_folder = args.valid_folder

gen_file = open(os.path.join(exp_root, valid_inference_folder + "text"))
asr_gen_file = open(os.path.join(exp_root, valid_inference_folder + "src_text"))
line_arr = [line for line in gen_file]
asr_line_arr = [line for line in asr_gen_file]
generate_entity_file(line_arr, asr_line_arr, output_file=os.path.join(exp_root,"result_valid.json"))

test_inference_folder = args.test_folder

gen_file = open(os.path.join(exp_root, test_inference_folder + "text"))
asr_gen_file = open(os.path.join(exp_root, test_inference_folder + "src_text"))
line_arr = [line for line in gen_file]
asr_line_arr = [line for line in asr_gen_file]
generate_entity_file(line_arr, asr_line_arr, output_file=os.path.join(exp_root,"result_test.json"))
