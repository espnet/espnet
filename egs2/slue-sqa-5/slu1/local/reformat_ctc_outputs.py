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
    def __init__(self, token_fn, log_dir, split="devel"):
        self.token_fn = token_fn
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.split = split

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
            if subtoken == "ANS":
                if part_of_entity:
                    end_time += dur
                    tstamp_tuple_lst.append((start_time, end_time))
                    part_of_entity = False
                else:
                    part_of_entity = True
                    start_time = np.sum(dur_lst[:idx])
                    end_time = start_time + dur
            elif part_of_entity:
                end_time += dur
        return tstamp_tuple_lst

    def get_char_outputs(self, frame_len):
        """
        Convert frame_level outputs to a sequence of words and timestamps
        """
        frame_level_op = read_lst(self.token_fn)
        res_dct = {}
        num_inconsistent = 0
        tot_cnt = 0
        for item in frame_level_op:
            utt_id = item.split(" ")[0]
            subtokens = item.split(" ")[1:]
            tot_cnt += 1
            subtoken_lst, dur_lst = self.deduplicate(subtokens, frame_len)
            if "ANS" in subtoken_lst:  # at least one entity
                res_dct[utt_id] = self.extract_entity_timestamps(subtoken_lst, dur_lst)
        save_dir = os.path.dirname(self.token_fn)
        save_json(os.path.join(self.log_dir, f"{self.split}_pred_stamps.json"), res_dct)


def main(token_fn, log_dir, split="test", frame_len=4e-2):
    obj = ReadCTCEmissions(token_fn, log_dir, split)
    obj.get_char_outputs(frame_len)


if __name__ == "__main__":
    Fire(main)
