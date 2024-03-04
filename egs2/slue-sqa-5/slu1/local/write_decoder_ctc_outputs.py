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
import re

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
        # self.nel_utt_lst = list(
        #     load_json(
        #         os.path.join("data", "nel_gt", f"{split}_all_word_alignments.json")
        #     ).keys()
        # )
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

    def extract_entity_ctc_timestamps(self, subtoken_lst, dur_lst):
        """
        extract start_time and end_time;
        starts at the start of FILL and ends at the end of SEP
        """
        tstamp_tuple_lst = []
        part_of_entity = False
        for idx, item_tuple in enumerate(zip(subtoken_lst, dur_lst)):
            subtoken, dur = item_tuple
            if subtoken == "ANS":
                # import pdb;pdb.set_trace()
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

    def extract_entity_timestamps(self, subtoken_lst):
        """
        extract start_time and end_time;
        starts at the start of FILL and ends at the end of SEP
        """
        tstamp_tuple_lst = []
        part_of_entity = False
        entity_token = ""
        for idx, subtoken in enumerate(subtoken_lst):
            if subtoken == "ANS":
                # import pdb;pdb.set_trace()
                if part_of_entity:
                    # import pdb;pdb.set_trace()
                    tstamp_tuple_lst.append((entity_token, 0, 0))
                    part_of_entity = False
                    entity_token = ""
                else:
                    part_of_entity = True
                    # start_time = np.sum(dur_lst[:idx])
                    # end_time = start_time + dur
            elif part_of_entity:
                # end_time += dur
                entity_token += subtoken
        return tstamp_tuple_lst

    def get_char_outputs(self, frame_len):
        """
        Convert frame_level outputs to a sequence of words and timestamps
        """
        frame_level_op = read_lst(self.token_fn)
        frame_level_op_ctc = read_lst(
            "/scratch/bbjs/arora1/espnet_slue_PR/espnet/egs2/slue-sqa-5/slu1/exp/slu_train_asr_wav2vec2_raw_en_bpe1000/decode_asr_full_path_slu_model_valid.acc.ave/test/token"
        )
        # lines = read_lst(self.og_data)[1:]
        # nel_utt_lst = [line.split("\t")[0] for line in lines]
        res_dct = {}
        num_inconsistent = 0
        tot_cnt = 0
        for i in range(len(frame_level_op)):
            item = frame_level_op[i]
            item_ctc = frame_level_op_ctc[i]
            utt_id = item.split(" ")[0]
            assert item_ctc.split(" ")[0] == utt_id
            subtokens = item.split(" ")[1:]
            subtokens_ctc = item_ctc.split(" ")[1:]
            # if utt_id in self.nel_utt_lst:  # process NEL corpus utterances only
            tot_cnt += 1
            subtoken_ctc_lst, dur_lst = self.deduplicate(subtokens_ctc, frame_len)
            if "ANS" in subtokens:  # at least one entity
                # cnt_fill = len(np.where(np.array(subtoken_lst) == "FILL")[0])
                # cnt_sep = len(np.where(np.array(subtoken_lst) == "SEP")[0])
                # if cnt_fill != cnt_sep:
                #     num_inconsistent += 1
                res_dct[utt_id] = self.extract_entity_timestamps(subtokens)
                # import pdb;pdb.set_trace()
                cnt_fill = len(np.where(np.array(subtoken_ctc_lst) == "ANS")[0])
                if len(res_dct[utt_id]) > 0:
                    try:
                        assert len(res_dct[utt_id]) == 1
                        ctc_out = "".join(subtoken_ctc_lst).replace("<blank>", "")
                        if ctc_out.count(res_dct[utt_id][0][0][1:-1]) == 0:
                            if cnt_fill == 2:
                                res_dct[utt_id] = self.extract_entity_ctc_timestamps(
                                    subtoken_ctc_lst, dur_lst
                                )
                            else:
                                print("ok")
                                res_dct[utt_id] = []
                        elif ctc_out.count(res_dct[utt_id][0][0][1:-1]) == 1:
                            # import pdb;pdb.set_trace()
                            add_count = ctc_out[
                                : ctc_out.index(res_dct[utt_id][0][0][1:-1])
                            ].count("▁")
                            find_count = 0
                            subtoken_ctc_lst1 = []
                            dur_lst1 = []
                            end_count = add_count + res_dct[utt_id][0][0].count("▁") - 1
                            first_found = False
                            last_found = False
                            for subtok_idx in range(len(subtoken_ctc_lst)):
                                k = subtoken_ctc_lst[subtok_idx]
                                if "▁" in k:
                                    find_count += 1
                                if find_count == add_count and first_found == False:
                                    subtoken_ctc_lst1.append("ANS")
                                    dur_lst1.append(0.0)
                                    first_found = True
                                elif find_count == end_count and last_found == False:
                                    subtoken_ctc_lst1.append("ANS")
                                    dur_lst1.append(0.0)
                                    last_found = True
                                if subtoken_ctc_lst[subtok_idx] == "ANS":
                                    subtoken_ctc_lst1.append("<blank>")
                                else:
                                    subtoken_ctc_lst1.append(
                                        subtoken_ctc_lst[subtok_idx]
                                    )
                                dur_lst1.append(dur_lst[subtok_idx])
                            cnt_fill = len(
                                np.where(np.array(subtoken_ctc_lst1) == "ANS")[0]
                            )
                            if cnt_fill < 2:
                                print("oookk")
                                subtoken_ctc_lst1.append("ANS")
                                dur_lst1.append(0.0)
                            elif cnt_fill > 2:
                                import pdb

                                pdb.set_trace()
                            res_dct[utt_id] = self.extract_entity_ctc_timestamps(
                                subtoken_ctc_lst1, dur_lst1
                            )
                            # import pdb;pdb.set_trace()
                        else:
                            # import pdb;pdb.set_trace()
                            ctc_start = [
                                m.start()
                                for m in re.finditer(
                                    res_dct[utt_id][0][0][1:-1], ctc_out
                                )
                            ]
                            dec_start = [
                                m.start()
                                for m in re.finditer(
                                    res_dct[utt_id][0][0][1:-1], "".join(subtokens)
                                )
                            ]
                            for dec_index in range(len(dec_start)):
                                if (
                                    "".join(subtokens)[: dec_start[dec_index]].count(
                                        "ANS"
                                    )
                                    == 1
                                ):
                                    if dec_index >= len(ctc_start):
                                        add_count = ctc_out[: ctc_start[-1]].count("▁")
                                    else:
                                        add_count = ctc_out[
                                            : ctc_start[dec_index]
                                        ].count("▁")
                                    break
                            # import pdb;pdb.set_trace()
                            find_count = 0
                            subtoken_ctc_lst1 = []
                            dur_lst1 = []
                            end_count = add_count + res_dct[utt_id][0][0].count("▁") - 1
                            first_found = False
                            last_found = False
                            for subtok_idx in range(len(subtoken_ctc_lst)):
                                k = subtoken_ctc_lst[subtok_idx]
                                if "▁" in k:
                                    find_count += 1
                                if find_count == add_count and first_found == False:
                                    subtoken_ctc_lst1.append("ANS")
                                    dur_lst1.append(0.0)
                                    first_found = True
                                elif find_count == end_count and last_found == False:
                                    subtoken_ctc_lst1.append("ANS")
                                    dur_lst1.append(0.0)
                                    last_found = True
                                if subtoken_ctc_lst[subtok_idx] == "ANS":
                                    subtoken_ctc_lst1.append("<blank>")
                                else:
                                    subtoken_ctc_lst1.append(
                                        subtoken_ctc_lst[subtok_idx]
                                    )
                                dur_lst1.append(dur_lst[subtok_idx])
                            # import pdb;pdb.set_trace()
                            cnt_fill = len(
                                np.where(np.array(subtoken_ctc_lst1) == "ANS")[0]
                            )
                            if cnt_fill < 2:
                                print("oookk")
                                subtoken_ctc_lst1.append("ANS")
                                dur_lst1.append(0.0)
                            elif cnt_fill > 2:
                                import pdb

                                pdb.set_trace()
                            res_dct[utt_id] = self.extract_entity_ctc_timestamps(
                                subtoken_ctc_lst1, dur_lst1
                            )
                            # import pdb;pdb.set_trace()
                    except:
                        import pdb

                        pdb.set_trace()

        # print(f"Tot samples: {len(self.nel_utt_lst)}")
        # print(f"Tot pred samples: {tot_cnt}")
        # print(f"Tot inconsistent samples: {num_inconsistent}")
        save_dir = os.path.dirname(self.token_fn)
        save_json(os.path.join(save_dir, f"{self.split}_pred_stamps.json"), res_dct)


def main(dir_name="token", split="test", frame_len=4e-2):
    obj = ReadCTCEmissions(dir_name, split)
    obj.get_char_outputs(frame_len)


if __name__ == "__main__":
    Fire(main)
