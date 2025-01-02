# pred_file=
word_dict={}
for line in open("/Users/siddhantarora/venv/bolt-samples/pytorch/Two_Channel_Label_Mono_update.csv"):
    line1=line.strip().split(",")
    if (("sw0"+line1[0]) not in word_dict):
        word_dict["sw0"+line1[0]]={}
    if float(line1[1]) not in word_dict["sw0"+line1[0]]:
        word_dict["sw0"+line1[0]][float(line1[1])]={}
    if float(line1[2]) not in word_dict["sw0"+line1[0]][float(line1[1])]:
        word_dict["sw0"+line1[0]][float(line1[1])][float(line1[2])]=line1
    else:
        print("error")
        import pdb;pdb.set_trace()

spk_dict={}
label_dict={}
for id1 in word_dict:
    seg_text_str=[]
    label_text_str=[]
    for j in word_dict[id1]:
        for j_end in word_dict[id1][j]:
            k=word_dict[id1][j][j_end]
            seg_text_str.append(k[-1])
            label_text_str.append(k[-2])
    spk_dict[id1]=seg_text_str
    label_dict[id1]=label_text_str

pred_file_arr=[]
str1="/Users/siddhantarora/venv/bolt-samples/pytorch/decode_asr_chunk_asr_model_valid.loss.ave_corrected_mix/test/logdir/output."
for k in range(1,5):
    pred_file_arr.append(open(str1+str(k)+"/1best_recog/text"))

# gt_file=
gt_arr=[k.strip().split()[1:] for k in open("/Users/siddhantarora/venv/bolt-samples/pytorch/data_2channel_mix/test/text")]

gt_id_arr=[k.split()[0] for k in open("/Users/siddhantarora/venv/bolt-samples/pytorch/data_2channel_mix/test/text")]
pred_arr=[]
spk_arr=[]
count=0
for pred_file in pred_file_arr:
    for line in pred_file:
        assert line.split()[0]==gt_id_arr[count]
        if gt_id_arr[count]=="sw04153":
            gt_arr[count]=["NA"]+gt_arr[count]
        # import pdb;pdb.set_trace()
        assert len(spk_dict[gt_id_arr[count]])==len(gt_arr[count])
        gt_arr[count]=label_dict[gt_id_arr[count]]
        spk_arr.append(spk_dict[gt_id_arr[count]])
        pred_index_arr=line.strip().split()[1:]
        if len(pred_index_arr)<len(gt_arr[count]):
            assert len(gt_arr[count])-len(pred_index_arr)==1
            # import pdb;pdb.set_trace()
            gt_arr[count]=gt_arr[count][:len(pred_index_arr)]
        elif len(pred_index_arr)>len(gt_arr[count]):
            assert len(pred_index_arr)-len(gt_arr[count])==1
            # import pdb;pdb.set_trace()
            pred_index_arr=pred_index_arr[:len(gt_arr[count])]
        count+=1
        for subline in pred_index_arr:
            line1=subline.split(",")
            pred_arr.append([float(k) for k in line1])

gt_arr_total=[]
count=-1
pred_arr_update=[]
for k in range(len(gt_arr)):
    for j in range(len(gt_arr[k])):
        label=gt_arr[k][j]
        spk_label=spk_arr[k][j]
        count+=1
        if label=="BC_1":
            continue
        elif label=="BC_2":
            continue
        elif label=="BC":
            if (spk_label!="A" and spk_label!="B"):
                continue
        elif label=="NA":
            continue
        gt_arr_total.append(label)
        pred_arr_update.append(pred_arr[count])
print(len(gt_arr_total))
print(len(pred_arr_update))
labels=["C","I","BC","T"]
score=0
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import numpy as np
gt_arr=np.array(gt_arr_total)
pred_arr=np.array(pred_arr_update)
count=0

pred_final_arr=[]
for i in range(len(pred_arr)):
    if pred_arr[i][0]>0.25:
        pred_final_arr.append("C")
    elif pred_arr[i][3]>0.4:
        pred_final_arr.append("BC")
    elif pred_arr[i][2]>0.4:
        pred_final_arr.append("I")
    elif pred_arr[i][4]>0.35:
        pred_final_arr.append("T")
    else:
        # import pdb;pdb.set_trace()
        pred_final_arr.append(np.argmax([pred_arr[i][0]]+list(pred_arr[i][2:])))
pred_final_arr=np.array(pred_final_arr)
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# cm = confusion_matrix(gt_arr, pred_final_arr, labels=labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                               display_labels=labels)
# disp.plot()
# plt.show()
count=0
for k in labels:
    print(k)
    x=f1_score(gt_arr==k, pred_final_arr==k, average="macro")
    print(classification_report(gt_arr==k, pred_final_arr==k))
    print(x)
    score+=x
    count+=1
print(score/4)