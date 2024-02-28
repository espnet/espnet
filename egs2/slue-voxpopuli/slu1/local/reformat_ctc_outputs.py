"""
Converts frame-level outputs into submission format:
Submission format: json file
{
    utt_1: [(start_time_1, end_time_1), ..., (start_time_k_1, end_time_k_1)],
    .
    .
    utt_N: [(start_time_1, end_time_1), ..., (start_time_k_N, end_time_k_N)],
}
"""

import json
import os

import numpy as np
from fire import Fire


# MY_DIR = "/share/data/lang/users/ankitap"
def read_lst(fname):
    with open(fname, "r") as f:
        lst_from_file = [line.strip() for line in f.readlines()]
    return lst_from_file


def write_to_file(write_str, fname):
    with open(fname, "w") as f:
        f.write(write_str)


def save_json(fname, dict_name):
    with open(fname, "w") as f:
        f.write(json.dumps(dict_name, indent=4))


def load_json(fname):
    data = json.loads(open(fname).read())
    return data


class ReadCTCEmissions:
    def __init__(self, dir_name, split="devel"):
        self.token_fn = dir_name
        self.split = split
        self.nel_utt_lst = list(
            load_json(
                os.path.join("data", "nel_gt", f"{split}_all_word_alignments.json")
            ).keys()
        )
        # self.og_data = os.path.join(
        #     MY_DIR,
        #     "nel_code_package/slue-toolkit/data/slue-voxpopuli_nel",
        #     f"slue-voxpopuli_nel_{split}.tsv"
        #     )

    def deduplicate(self, output_tokens, frame_len):
        """
        Get a list of unique token sequence
        """
        concise_lst = [output_tokens[0]]
        dur_lst = []
        curr_length = 1
        for char in output_tokens[1:]:
            if char != concise_lst[-1]:
                concise_lst.append(char)
                dur_lst.append(frame_len * curr_length)
                curr_length = 0
            curr_length += 1
        dur_lst.append(frame_len * curr_length)
        assert len(concise_lst) == len(dur_lst)
        assert np.round(np.sum(dur_lst), 1) == np.round(
            frame_len * len(output_tokens), 1
        )
        return concise_lst, dur_lst

    def extract_entity_timestamps(self, subtoken_lst, dur_lst):
        """
        extract start_time and end_time;
        starts at the start of FILL and ends at the end of SEP
        """
        tstamp_tuple_lst = []
        part_of_entity = False
        for idx, item_tuple in enumerate(zip(subtoken_lst, dur_lst)):
            subtoken, dur = item_tuple
            if subtoken == "FILL":
                part_of_entity = True
                start_time = np.sum(dur_lst[:idx])
                end_time = start_time + dur
            elif subtoken == "SEP":
                if part_of_entity:
                    end_time += dur
                    tstamp_tuple_lst.append((start_time, end_time))
                part_of_entity = False
            elif part_of_entity:
                end_time += dur
        return tstamp_tuple_lst

    def get_char_outputs(self, frame_len):
        """
        Convert frame_level outputs to a sequence of words and timestamps
        """
        frame_level_op = read_lst(self.token_fn)
        # lines = read_lst(self.og_data)[1:]
        # nel_utt_lst = [line.split("\t")[0] for line in lines]
        res_dct = {}
        num_inconsistent = 0
        tot_cnt = 0
        for item in frame_level_op:
            utt_id = "_".join(item.split(" ")[0].split("_")[1:])
            subtokens = item.split(" ")[1:]
            if utt_id in self.nel_utt_lst:  # process NEL corpus utterances only
                tot_cnt += 1
                subtoken_lst, dur_lst = self.deduplicate(subtokens, frame_len)
                if (
                    "FILL" in subtoken_lst and "SEP" in subtoken_lst
                ):  # at least one entity
                    cnt_fill = len(np.where(np.array(subtoken_lst) == "FILL")[0])
                    cnt_sep = len(np.where(np.array(subtoken_lst) == "SEP")[0])
                    if cnt_fill != cnt_sep:
                        num_inconsistent += 1
                    res_dct[utt_id] = self.extract_entity_timestamps(
                        subtoken_lst, dur_lst
                    )
        # print(f"Tot samples: {len(self.nel_utt_lst)}")
        # print(f"Tot pred samples: {tot_cnt}")
        # print(f"Tot inconsistent samples: {num_inconsistent}")
        save_dir = os.path.dirname(self.token_fn)
        save_json(os.path.join(save_dir, f"{self.split}_pred_stamps.json"), res_dct)


def main(dir_name="token", split="devel", frame_len=4e-2):
    obj = ReadCTCEmissions(dir_name, split)
    obj.get_char_outputs(frame_len)


if __name__ == "__main__":
    Fire(main)
