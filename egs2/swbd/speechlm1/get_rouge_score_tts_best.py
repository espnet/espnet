from nltk.translate.meteor_score import meteor_score
import numpy as np
import evaluate
import nltk
nltk.download('wordnet')
import sys

references = []
reference_id=[]
reference_text=open(sys.argv[1])
reference_dict={}
for line in reference_text:
    reference_dict[line.strip().split()[0]]=" ".join(line.strip().split()[1:]).lower()
    # reference_id.append(line.strip().split()[0])

candidates = []
candidate_id=[]
file=open(sys.argv[2])
best_dict={}
for line in file:
    best_dict[line.strip()]=1
# line_arr1=[]
candidate_text=open(sys.argv[3])
id1_arr=[]
for line in candidate_text:
    if line.split()[0] in best_dict:
        candidates.append(" ".join(line.strip().split()[1:]))
        id1="_".join(line.strip().split()[0].split("_")[5:-1])
        id1_arr.append(id1)
        references.append(reference_dict[id1])


print(len(references))
print(len(candidates))
score_arr=[]
summ_metrics = evaluate.combine(["rouge", "meteor"])
import pdb;pdb.set_trace()
result = summ_metrics.compute(
    predictions=candidates,
    references=references,
)
rouge = f"{result['rouge1']*100} {result['rouge2']*100} {result['rougeL']*100}"
mtr = f"{result['meteor']*100}"
print(f"RESULT {rouge} {mtr}")
import pdb;pdb.set_trace()
summ_metrics = evaluate.combine(["rouge", "meteor"])
rouge1_dict={}
rouge2_dict={}
rougeL_dict={}
meteor_dict={}
for i, (cand, ref) in enumerate(zip(candidates, references)):
    result = summ_metrics.compute(
        predictions=[cand],
        references=[ref],
    )
    rouge1_dict[id1_arr[i]]=round(result['rouge1']*100,2)
    rouge2_dict[id1_arr[i]]=round(result['rouge2']*100,2)
    rougeL_dict[id1_arr[i]]=round(result['rougeL']*100,2)
    meteor_dict[id1_arr[i]]=round(result['meteor']*100,2)
import pickle
with open("rouge1_cot_dict.pkl", "wb") as f:
    pickle.dump(rouge1_dict, f)
with open("rouge2_cot_dict.pkl", "wb") as f:
    pickle.dump(rouge2_dict, f)
with open("rougeL_cot_dict.pkl", "wb") as f:
    pickle.dump(rougeL_dict, f)
with open("meteor_cot_dict.pkl", "wb") as f:
    pickle.dump(meteor_dict, f)

# rouge = f"{result['rouge1']*100} {result['rouge2']*100} {result['rougeL']*100}"
# mtr = f"{result['meteor']*100}"
# print(f"RESULT {rouge} {mtr}")