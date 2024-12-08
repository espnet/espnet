def clean_line(k):
    k = k.replace("[noise]", "")
    k = k.replace("[silence]", "")
    k = k.replace("[vocalized-noise]", "")
    k = k.replace("]", "")
    k = k.replace("[", "")
    k = k.replace("-", " ")
    return k


def create_dialog_sort(sent_file_A, sent_file_B):
    sentence_dict = {}
    for line in sent_file_A:
        str1 = clean_line((" ".join(line.split()[3:])))
        if str1.strip() == "":
            continue
        if float(line.split()[1]) in sentence_dict:
            print("joint start")
            print(line)
            sentence_dict[float(line.split()[1]) + 1e-6] = [
                "A",
                float(line.split()[2]),
                line.split()[0],
            ]
        else:
            sentence_dict[float(line.split()[1])] = [
                "A",
                float(line.split()[2]),
                line.split()[0],
            ]
    for line in sent_file_B:
        str1 = clean_line((" ".join(line.split()[3:])))
        if str1.strip() == "":
            continue
        if float(line.split()[1]) in sentence_dict:
            print("joint_start")
            print(line)
            sentence_dict[float(line.split()[1]) + 1e-6] = [
                "B",
                float(line.split()[2]),
                line.split()[0],
            ]
        else:
            sentence_dict[float(line.split()[1])] = [
                "B",
                float(line.split()[2]),
                line.split()[0],
            ]
    sentence_dict_sort = dict(sorted(sentence_dict.items()))
    return sentence_dict_sort


def remove_backchannel(file_name, backchannel_dict, word_file):
    line_file_filtered_A = []
    line_backchannel_A = []
    line_file_unfiltered = [line for line in word_file]
    count = 0
    if file_name not in backchannel_dict:
        while count < len(line_file_unfiltered):
            line1 = line_file_unfiltered[count].split()
            count += 1
            word_val = clean_line(line1[3].strip())
            if word_val != "":
                line1_better = [line1[0], line1[1], line1[2], word_val]
                line_file_filtered_A.append(line1_better)
        return line_file_filtered_A, []
    for val in backchannel_dict[file_name]:
        while count < len(line_file_unfiltered):
            line1 = line_file_unfiltered[count].split()
            count += 1
            if (line1[0]) == val:
                break
            else:
                word_val = clean_line(line1[3].strip())
                if word_val != "":
                    line1_better = [line1[0], line1[1], line1[2], word_val]
                    line_file_filtered_A.append(line1_better)
        if (line1[0]) in backchannel_dict[file_name]:
            for sub_val in backchannel_dict[file_name][line1[0]]:
                found_1 = False
                found_2 = False
                while count < len(line_file_unfiltered):
                    if sub_val[0] == float(line1[1]):
                        found_1 = True
                    if sub_val[1] == float(line1[2]):
                        found_2 = True
                    if found_1 and found_2:
                        line_backchannel_A.append(line1)
                        found_1 = False
                        found_2 = False
                        break
                    elif found_1:
                        line_backchannel_A.append(line1)
                    elif found_2:
                        if sub_val[0] != sub_val[1]:
                            print("error")
                            import pdb

                            pdb.set_trace()
                    else:
                        word_val = clean_line(line1[3].strip())
                        if word_val != "":
                            line1_better = [line1[0], line1[1], line1[2], word_val]
                            line_file_filtered_A.append(line1_better)
                    line1 = line_file_unfiltered[count].split()
                    count += 1
    while count < len(line_file_unfiltered):
        line1 = line_file_unfiltered[count].split()
        count += 1
        word_val = clean_line(line1[3].strip())
        if word_val != "":
            line1_better = [line1[0], line1[1], line1[2], word_val]
            line_file_filtered_A.append(line1_better)
    return line_file_filtered_A, line_backchannel_A


def clean_dialogue(word_file_A, word_file_B, sentence_dict_sort):
    count_A = 0
    count_B = 0
    prev_speaker = "NA"
    sentence_dict_sort_clean = {}
    for k in sentence_dict_sort:
        if "A" == sentence_dict_sort[k][0]:
            if count_A >= len(word_file_A):
                continue
            while float(word_file_A[count_A][2]) <= sentence_dict_sort[k][1]:
                if word_file_A[count_A][0] != sentence_dict_sort[k][2]:
                    import pdb

                    pdb.set_trace()
                if k not in sentence_dict_sort_clean:
                    sentence_dict_sort_clean[k] = sentence_dict_sort[k] + [
                        float(word_file_A[count_A][1]),
                        float(word_file_A[count_A][2]),
                    ]
                else:
                    if sentence_dict_sort_clean[k][-2] > float(word_file_A[count_A][1]):
                        sentence_dict_sort_clean[k][-2] = float(
                            word_file_A[count_A][1]
                        )  # Update start time
                    if sentence_dict_sort_clean[k][-1] < float(word_file_A[count_A][2]):
                        sentence_dict_sort_clean[k][-1] = float(
                            word_file_A[count_A][2]
                        )  # Update end time
                count_A += 1
                if count_A >= len(word_file_A):
                    break
        elif "B" == sentence_dict_sort[k][0]:
            if count_B >= len(word_file_B):
                continue
            while float(word_file_B[count_B][2]) <= sentence_dict_sort[k][1]:
                if word_file_B[count_B][0] != sentence_dict_sort[k][2]:
                    import pdb

                    pdb.set_trace()
                if k not in sentence_dict_sort_clean:
                    sentence_dict_sort_clean[k] = sentence_dict_sort[k] + [
                        float(word_file_B[count_B][1]),
                        float(word_file_B[count_B][2]),
                    ]
                else:
                    if sentence_dict_sort_clean[k][-2] > float(word_file_B[count_B][1]):
                        sentence_dict_sort_clean[k][-2] = float(
                            word_file_B[count_B][1]
                        )  # Update start time
                    if sentence_dict_sort_clean[k][-1] < float(word_file_B[count_B][2]):
                        sentence_dict_sort_clean[k][-1] = float(
                            word_file_B[count_B][2]
                        )  # Update end time
                count_B += 1
                if count_B >= len(word_file_B):
                    break
        else:
            print("error")
            import pdb

            pdb.set_trace()
    while count_A < len(word_file_A):
        if float(word_file_A[count_A][2]) != sentence_dict_sort[k][1]:
            print("error")
            import pdb

            pdb.set_trace()
        else:
            break
    while count_B < len(word_file_B):
        if float(word_file_B[count_B][2]) != sentence_dict_sort[k][1]:
            print("error")
            import pdb

            pdb.set_trace()
        else:
            break
    return sentence_dict_sort_clean


def remove_overlapping_IPUs(sentence_dict):
    sentence_dict_clean = {}
    for k in sentence_dict:
        sentence_dict_clean[sentence_dict[k][-2]] = [
            sentence_dict[k][0],
            sentence_dict[k][-1],
            sentence_dict[k][2],
        ]
    sentence_dict_sort = dict(sorted(sentence_dict_clean.items()))
    sentence_dict_remove_intersect = {}
    prev_k = "NA"
    for k in sentence_dict_sort:
        if prev_k == "NA":
            sentence_dict_remove_intersect[k] = sentence_dict_sort[k]
        else:
            if (sentence_dict_sort[prev_k][1] >= k) and (
                sentence_dict_sort[prev_k][1] >= sentence_dict_sort[k][1]
            ):
                print("completely overlapped")
                print([prev_k] + sentence_dict_sort[prev_k])
                print([k] + sentence_dict_sort[k])
                continue
            else:
                sentence_dict_remove_intersect[k] = sentence_dict_sort[k]
        prev_k = k
    return sentence_dict_remove_intersect


def create_backchannel_sort_arr(backchannel_dict, file_name):
    backchannel_spk = backchannel_dict[file_name]
    backchannel_sort_spk = {}
    for k in backchannel_spk:
        for backchannel_time in backchannel_spk[k]:
            print(backchannel_time)
            if backchannel_time[0] in backchannel_sort_spk:
                print("error")
                exit()
            backchannel_sort_spk[backchannel_time[0]] = backchannel_time
    backchannel_sort_spk = dict(sorted(backchannel_sort_spk.items()))
    backchannel_sort_arr = [backchannel_sort_spk[k] for k in backchannel_sort_spk]
    return backchannel_sort_arr
