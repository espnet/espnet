import csv

import pandas as pd
from local.turn_take_utils import (
    clean_dialogue,
    create_backchannel_sort_arr,
    create_dialog_sort,
    remove_backchannel,
)

chunk_length = 0.04
min_start = 0.2


def get_label(start_chunk, end_chunk, label_arr, arr_count):
    label = False
    if arr_count < len(label_arr):
        if start_chunk >= label_arr[arr_count][1]:
            if start_chunk == min_start:
                arr_count += 1
                return get_label(start_chunk, end_chunk, label_arr, arr_count)
            else:
                print("error")
                import pdb

                pdb.set_trace()
        elif end_chunk >= label_arr[arr_count][1]:
            arr_count += 1
            return get_label(start_chunk, end_chunk, label_arr, arr_count)
        elif (end_chunk + chunk_length) >= label_arr[arr_count][0]:
            label = True
    return label, arr_count


def get_chunk_dict(
    speaker, backchannel, word_lines, file_id, last_time, chunk_dict, local_start_time
):
    backchannel_count = 0
    activity_count = 0
    start_time = float("{:.2f}".format(local_start_time))
    end_time = float("{:.2f}".format(last_time))
    total_chunk = int((end_time - start_time) / chunk_length)
    activity_arr = [[float(k[1]), float(k[2])] for k in word_lines]
    for i in range(total_chunk):
        start_chunk = start_time + i * chunk_length
        end_chunk = start_time + (i + 1) * chunk_length
        if start_chunk >= end_time:
            import pdb

            pdb.set_trace()
        word_key = file_id + "_" + str(start_chunk) + "_" + str(end_chunk)
        if word_key not in chunk_dict:
            chunk_dict[word_key] = {}
            chunk_dict[word_key]["file_id"] = file_id
            chunk_dict[word_key]["chunk_start"] = float("{:.2f}".format(start_chunk))
            chunk_dict[word_key]["chunk_end"] = float("{:.2f}".format(end_chunk))
            chunk_dict[word_key]["label_A"] = "NA"
            chunk_dict[word_key]["label_B"] = "NA"
        bc_label, backchannel_count = get_label(
            start_chunk, end_chunk, backchannel, backchannel_count
        )
        activity_label, activity_count = get_label(
            start_chunk, end_chunk, activity_arr, activity_count
        )
        if speaker == "A":
            if bc_label:
                chunk_dict[word_key]["label_A"] = "BC"
            elif activity_label:
                chunk_dict[word_key]["label_A"] = "IPU"
        else:
            if bc_label:
                chunk_dict[word_key]["label_B"] = "BC"
            elif activity_label:
                chunk_dict[word_key]["label_B"] = "IPU"
    return chunk_dict


backchannel_dict = {}
df = pd.read_csv("local/backchannels.csv")
for line in df.values:
    line_file = "-".join(line[3].split("-")[:-1])
    if line_file not in backchannel_dict:
        backchannel_dict[line_file] = {}
    if line[3] not in backchannel_dict[line_file]:
        backchannel_dict[line_file][line[3]] = []
    backchannel_dict[line_file][line[3]].append([float(line[0]), float(line[1])])


def resort_dialogue(sentence_dict):
    sentence_dict_clean = {}
    for k in sentence_dict:
        sentence_dict_clean[sentence_dict[k][-2]] = [
            sentence_dict[k][0],
            sentence_dict[k][-1],
            sentence_dict[k][2],
        ]
    sentence_dict_sort = dict(sorted(sentence_dict_clean.items()))
    return sentence_dict_sort


def get_duration():
    file = open("sox_duration.txt")
    line_arr = [line for line in file]
    line_arr1 = []
    for line in line_arr:
        if "Length (seconds):" in line:
            line_arr1.append(float(line.strip().split()[-1]))

    print(len(line_arr1))

    line_arr2 = []
    file1 = open("sox_duration.sh")
    id_dict = {}
    for line in file1:
        id1 = line.split()[1].split("/")[-1].split(".")[0]
        id_dict[id1] = line_arr1[len(id_dict)]

    return id_dict


field_names = ["file_id", "chunk_start", "chunk_end", "label_A", "label_B"]
dir_dict = {
    "train": "Train_Two_Channel_Label.csv",
    "val": "Val_Two_Channel_Label.csv",
    "test": "Test_Two_Channel_Label.csv",
}
for x in dir_dict:
    csvfile = open(dir_dict[x], "w")
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    duration_dict = get_duration()
    writer.writeheader()
    for line in open("turn_take_splits/" + x + ".txt"):
        file_id = line.strip()
        print("----ok1---")
        print(file_id)
        print("----ok1---")
        dir_id = file_id[:2]
        file_name_A = "sw" + file_id + "A-ms98-a"
        file_name_B = "sw" + file_id + "B-ms98-a"
        word_file_A = open(
            "data/local/train/swb_ms98_transcriptions/"
            + dir_id
            + "/"
            + file_id
            + "/"
            + file_name_A
            + "-word.text"
        )
        word_file_B = open(
            "data/local/train/swb_ms98_transcriptions/"
            + dir_id
            + "/"
            + file_id
            + "/"
            + file_name_B
            + "-word.text"
        )
        sent_file_A = open(
            "data/local/train/swb_ms98_transcriptions/"
            + dir_id
            + "/"
            + file_id
            + "/"
            + file_name_A
            + "-trans.text"
        )
        sent_file_B = open(
            "data/local/train/swb_ms98_transcriptions/"
            + dir_id
            + "/"
            + file_id
            + "/"
            + file_name_B
            + "-trans.text"
        )
        sentence_dict_sort = create_dialog_sort(sent_file_A, sent_file_B)
        line_file_filtered_A, line_backchannel_A = remove_backchannel(
            file_name_A, backchannel_dict, word_file_A
        )
        line_file_filtered_B, line_backchannel_B = remove_backchannel(
            file_name_B, backchannel_dict, word_file_B
        )
        sentence_dict_sort_clean = clean_dialogue(
            line_file_filtered_A, line_file_filtered_B, sentence_dict_sort
        )
        sentence_dict_sort_final = resort_dialogue(sentence_dict_sort_clean)
        if file_name_A in backchannel_dict:
            backchannel_A = create_backchannel_sort_arr(backchannel_dict, file_name_A)
        else:
            backchannel_A = []
        if file_name_B in backchannel_dict:
            backchannel_B = create_backchannel_sort_arr(backchannel_dict, file_name_B)
        else:
            backchannel_B = []
        last_time = duration_dict["sw0" + file_id]
        local_start_time = min_start
        chunk_dict = {}
        get_chunk_dict(
            "A",
            backchannel_A,
            line_file_filtered_A,
            file_id,
            last_time,
            chunk_dict,
            local_start_time,
        )
        get_chunk_dict(
            "B",
            backchannel_B,
            line_file_filtered_B,
            file_id,
            last_time,
            chunk_dict,
            local_start_time,
        )
        for chunk in chunk_dict:
            writer.writerow(chunk_dict[chunk])
