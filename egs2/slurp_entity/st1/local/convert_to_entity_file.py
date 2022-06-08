import json
import sys
import argparse
import os


def generate_entity_file(line_arr, output_file="result_test.json"):
    fp = open(output_file, "w")
    for line in line_arr:
        ent_final_arr = []
        if not(line.strip().split("\t")[0]=="<na>"):
            # print("yes")
            entity_names_arr = line.strip().split("\t")[0].split("SEP")
            # print(entity_names_arr)
            for entity in entity_names_arr:
                if len(entity.split("FILL")) != 2:
                    continue
                ent_type = entity.split("FILL")[0].strip()
                ent_val = entity.split("FILL")[1].strip().replace("'", "'")
                dict1 = {}
                dict1["type"] = ent_type
                dict1["filler"] = ent_val
                ent_final_arr.append(dict1)
        file_name = line.strip().split("\t")[1].split("_")[-1].replace(")", "")
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
test_inference_folder = args.test_folder

gen_file = open(os.path.join(exp_root, test_inference_folder + "score_bleu/hyp.trn.org"))
line_arr = [line for line in gen_file]
generate_entity_file(line_arr)

valid_inference_folder = args.valid_folder

gen_file = open(os.path.join(exp_root, valid_inference_folder + "score_bleu/hyp.trn.org"))
line_arr = [line for line in gen_file]
generate_entity_file(line_arr, output_file="result_valid.json")
