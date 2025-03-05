import sys

import numpy as np
from sklearn.metrics import roc_auc_score

word_dict = {}
for line in open("data/Test_Two_Channel_Label_Mono.csv"):
    line1 = line.strip().split(",")
    if ("sw0" + line1[0]) not in word_dict:
        word_dict["sw0" + line1[0]] = {}
    if float(line1[1]) not in word_dict["sw0" + line1[0]]:
        word_dict["sw0" + line1[0]][float(line1[1])] = {}
    if float(line1[2]) not in word_dict["sw0" + line1[0]][float(line1[1])]:
        word_dict["sw0" + line1[0]][float(line1[1])][float(line1[2])] = line1
    else:
        print("error")

spk_dict = {}
label_dict = {}
for id1 in word_dict:
    seg_text_str = []
    label_text_str = []
    for j in word_dict[id1]:
        for j_end in word_dict[id1][j]:
            k = word_dict[id1][j][j_end]
            seg_text_str.append(k[-1])
            label_text_str.append(k[-2])
    spk_dict[id1] = seg_text_str
    label_dict[id1] = label_text_str

pred_file_arr = []
exp_path_str1 = sys.argv[1]
for k in range(1, 5):
    pred_file_arr.append(open(exp_path_str1 + str(k) + "/1best_recog/text"))

# gt_file=
gt_arr = [k.strip().split()[1:] for k in open("data/test/text")]

gt_id_arr = [k.split()[0] for k in open("data/test/text")]
pred_arr = []
spk_arr = []
count = 0
for pred_file in pred_file_arr:
    for line in pred_file:
        assert line.split()[0] == gt_id_arr[count]
        assert len(spk_dict[gt_id_arr[count]]) == len(gt_arr[count])
        assert gt_arr[count] == label_dict[gt_id_arr[count]]
        spk_arr.append(spk_dict[gt_id_arr[count]])
        pred_index_arr = line.strip().split()[1:]
        if len(pred_index_arr) < len(gt_arr[count]):
            assert len(gt_arr[count]) - len(pred_index_arr) == 1
            gt_arr[count] = gt_arr[count][: len(pred_index_arr)]
        elif len(pred_index_arr) > len(gt_arr[count]):
            assert len(pred_index_arr) - len(gt_arr[count]) == 1
            pred_index_arr = pred_index_arr[: len(gt_arr[count])]
        count += 1
        for subline in pred_index_arr:
            line1 = subline.split(",")
            pred_arr.append([float(k) for k in line1])

gt_arr_total = []
count = -1
pred_arr_update = []
for k in range(len(gt_arr)):
    for j in range(len(gt_arr[k])):
        label = gt_arr[k][j]
        spk_label = spk_arr[k][j]
        count += 1
        if label == "BC_1":
            continue
        elif label == "BC_2":
            continue
        elif label == "BC":
            if spk_label != "A" and spk_label != "B":
                continue
        gt_arr_total.append(label)
        pred_arr_update.append(pred_arr[count])
print(len(gt_arr_total))
print(len(pred_arr_update))

labels = ["C", "NA", "I", "BC", "T"]
score = 0

gt_arr = np.array(gt_arr_total)
pred_arr = np.array(pred_arr_update)
count = 0
for k in labels:
    print(k)
    x = roc_auc_score(gt_arr == k, pred_arr[:, count])
    print(x)
    score += x
    count += 1
print(score / 5)
