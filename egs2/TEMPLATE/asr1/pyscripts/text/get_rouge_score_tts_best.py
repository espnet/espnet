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

candidates = []
candidate_id=[]
file=open(sys.argv[2])
best_dict={}
for line in file:
    best_dict[line.strip()]=1
candidate_text=open(sys.argv[3])
id1_arr=[]
for line in candidate_text:
    if line.split()[0] in best_dict:
        candidates.append(" ".join(line.strip().split()[1:]))
        id1="_".join(line.strip().split()[0].split("_")[5:-1])
        id1_arr.append(id1)
        references.append(reference_dict[id1])


score_arr=[]
summ_metrics = evaluate.combine(["rouge", "meteor"])
result = summ_metrics.compute(
    predictions=candidates,
    references=references,
)
rouge = f"{result['rouge1']*100} {result['rouge2']*100} {result['rougeL']*100}"
mtr = f"{result['meteor']*100}"
print(f"RESULT {rouge} {mtr}")

