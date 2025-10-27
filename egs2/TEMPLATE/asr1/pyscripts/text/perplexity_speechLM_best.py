import os
import csv
import sys
response_arr=[]
key_arr=[]
file=open(sys.argv[1])
best_dict={}
for line in file:
    best_dict[line.strip()]=1
id_arr=[]
# for i in range(1,2):
file=open(sys.argv[2])
for line in file:
    if line.split()[0] in best_dict:
        if len(line.strip().split())<2:
            continue
        response_arr.append(" ".join(line.strip().split()[1:]))
        id_arr.append(line.strip().split("_sample")[0])

import evaluate
print(len(response_arr))
perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(model_id='gpt2',predictions=response_arr)
# import pdb;pdb.set_trace()
print(round(results["mean_perplexity"], 2)) # doctest: +SKIP
# perplexity_dict={}
# for i in range(len(results['perplexities'])):
#     perplexity_dict[id_arr[i].replace("codec_ssl_cot_full_utt2spk_","")]=round(results['perplexities'][i],2)
# import pickle
# with open("perplexity_cot_dict.pkl", "wb") as f:
#     pickle.dump(perplexity_dict, f)
