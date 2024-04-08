import argparse
import json
import os
import sys

from espnet2.utils.types import str2bool


def generate_entity_file(line_arr, output_file="result_test.json", token_type_bpe=True):
    fp = open(output_file, "w")
    for line in line_arr:
        scenario = line.strip().split("\t")[0].split("_")[0].replace("in:", "")
        action = "_".join(line.strip().split("\t")[0].split()[0].split("_")[1:])
        if not token_type_bpe:
            line = line.replace(" ", "▁")
        entity_names_arr = line.strip().split("SEP")[1:-1]
        ent_final_arr = []
        for entity in entity_names_arr:
            if len(entity.split("FILL")) != 2:
                continue
            ent_type = entity.split("FILL")[0].strip().replace("sl:", "")
            if not token_type_bpe:
                ent_type = ent_type.replace("▁", " ").strip()
            ent_val = entity.split("FILL")[1].strip().replace("'", "'")
            dict1 = {}
            dict1["type"] = ent_type
            dict1["filler"] = ent_val
            ent_final_arr.append(dict1)
        if len(line.strip().split("\t")) == 1:
            print(line)
            file_name = line.strip().split("\t")[0].split("_")[-1].replace(")", "")
        else:
            file_name = line.strip().split("\t")[1].split("_")[-1].replace(")", "")
        file_name = "audio-" + file_name + ".flac"
        write_dict = {}
        write_dict["text"] = ""
        write_dict["scenario"] = scenario
        write_dict["action"] = action
        write_dict["entities"] = ent_final_arr
        write_dict["file"] = file_name
        json.dump(write_dict, fp)
        fp.write("\n")


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
parser.add_argument(
    "--valid_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/devel/",
    help="Directory inside exp_root containing inference on valid set",
)
parser.add_argument(
    "--test_folder",
    default="decode_asr_asr_model_valid.acc.ave_10best/test/",
    help="Directory inside exp_root containing inference on test set",
)
parser.add_argument(
    "--token_type_bpe",
    type=str2bool,
    default=True,
    help="Whether text is encoded in BPE units",
)

args = parser.parse_args()

exp_root = args.exp_root
valid_inference_folder = args.valid_folder
test_inference_folder = args.test_folder

gen_file = open(os.path.join(exp_root, test_inference_folder + "score_wer/hyp.trn"))
line_arr = [line for line in gen_file]
generate_entity_file(
    line_arr,
    output_file=os.path.join(exp_root, "result_test.json"),
    token_type_bpe=args.token_type_bpe,
)
