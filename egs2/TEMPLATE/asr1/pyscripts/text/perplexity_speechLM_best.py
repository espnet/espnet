import os
import csv
import sys
import evaluate

response_arr=[]
key_arr=[]
file=open(sys.argv[1])
best_dict={}
for line in file:
    best_dict[line.strip()]=1
id_arr=[]

file=open(sys.argv[2])
for line in file:
    if line.split()[0] in best_dict:
        if len(line.strip().split())<2:
            continue
        response_arr.append(" ".join(line.strip().split()[1:]))
        id_arr.append(line.strip().split("_sample")[0])

perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(model_id='gpt2',predictions=response_arr)
print(round(results["mean_perplexity"], 2))

