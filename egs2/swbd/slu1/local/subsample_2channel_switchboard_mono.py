import random

dir_arr = [
    "Train_Two_Channel_Label_Mono",
    "Val_Two_Channel_Label_Mono",
]
for x in dir_arr:
    csvfile = open(x + ".csv")
    csvfile_write = open(x + "_subsample.csv", "w")
    line_arr = [line for line in csvfile]

    turn_dict = {}
    turn_subsample_dict = {}
    prev_line1 = None
    current_key = None
    for line in line_arr:
        line1 = line.strip().split(",")
        if line1[-2] == "BC_1":
            continue
        elif line1[-2] == "BC_2":
            continue
        elif line1[-2] == "BC":
            if line1[-1] != "A" and line1[-1] != "B":
                continue
        if line1[-2] not in turn_dict:
            turn_dict[line1[-2]] = []
            turn_subsample_dict[line1[-2]] = {}
        if prev_line1 is None:
            turn_subsample_dict[line1[-2]][line1[0] + "_" + line1[1]] = []
            current_key = line1[0] + "_" + line1[1]
        else:
            if prev_line1[0] != line1[0]:
                turn_subsample_dict[line1[-2]][line1[0] + "_" + line1[1]] = []
                current_key = line1[0] + "_" + line1[1]
            elif line1[-2] != prev_line1[-2]:
                turn_subsample_dict[line1[-2]][line1[0] + "_" + line1[1]] = []
                current_key = line1[0] + "_" + line1[1]
        if line1[-2] == "T":
            csvfile_write.write(line)
        turn_dict[line1[-2]].append(line)
        turn_subsample_dict[line1[-2]][current_key].append(line)
        prev_line1 = line1

    count_subsample = len(turn_dict["T"])

    index_count = 0
    for k in turn_subsample_dict:
        if k == "T":
            continue
        subsample_dialogue_ratio = len(turn_dict[k]) / count_subsample
        index_count += 1
        new_count = 0
        subsample_arr_total = []
        for j in turn_subsample_dict[k]:
            a1 = max(1, len(turn_subsample_dict[k][j]) / subsample_dialogue_ratio + 1)
            if k == "BC" or k == "I":
                if int(a1) == 1:
                    subsample_arr = [turn_subsample_dict[k][j][0]]
                else:
                    subsample_arr = [turn_subsample_dict[k][j][0]] + random.sample(
                        turn_subsample_dict[k][j][1:], int(a1) - 1
                    )
                subsample_arr_total += subsample_arr
                new_count += int(a1)
            else:
                subsample_arr = random.sample(turn_subsample_dict[k][j], int(a1))
                subsample_arr_total += subsample_arr
                new_count += int(a1)
        subsample_arr = subsample_arr_total
        for line in subsample_arr:
            csvfile_write.write(line)
