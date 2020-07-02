#! /bin/python

# Copyright 2020 Carnegie Mellon University (Roshan Sharma)
# Copyright 2020 Carnegie Mellon University (Xuandi Fu)

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='SLU1_Baseline')
parser.add_argument('--expdir', type=str, default='../exp/train_pytorch_train_rnn_no_preprocess/decode_test_decode',
                    help='location of the decoded result')
parser.add_argument('--dict', type=str, default='../data/lang_1char/train_units.txt',
                    help='path of the dict file')


def read_test_result(path):
    num_utter = 0
    len_mismatch = 0
    pred4 = 0
    content_mismatch = 0
    rec_tokenid = []
    tokenid = []
    filepath = os.path.join(path,'data.json')
    print(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
        for key, outputs in data['utts'].items():
            num_utter += 1
            rec_tokenid_len = len(outputs['output'][0]['rec_tokenid'].split(' '))
            tokenid_len = len(outputs['output'][0]['tokenid'].split(' '))

            rec_tokenid.extend(outputs['output'][0]['rec_tokenid'].split(' ')[:3])
            tokenid.extend(outputs['output'][0]['tokenid'].split(' ')[:3])

            if rec_tokenid_len != tokenid_len:
                len_mismatch += 1
            elif rec_tokenid_len == tokenid_len:
                if outputs['output'][0]['rec_tokenid'] != outputs['output'][0]['tokenid']:
                    content_mismatch += 1

            if tokenid_len == 3 and rec_tokenid_len == 4:
                pred4 += 1

    return num_utter, tokenid, rec_tokenid

def acc_score(tokenid, rec_tokenid,num_utter,file):
    target = []

    with open(file, 'r') as f:

        for i in f.readlines():
            target.append(i.split(' ')[0])
    classrep = classification_report(tokenid, rec_tokenid, labels=np.arange(0, len(target)), target_names=target)
    print(classrep)
    acc_score = accuracy_score(tokenid, rec_tokenid)
    print("accuracy score: " + str(accuracy_score(tokenid, rec_tokenid)))

    accurate_utter = 0
    for i in range(int(len(tokenid) / 3)):
        if tokenid[i * 3:(i + 1) * 3] == rec_tokenid[i * 3:(i + 1) * 3]:
            accurate_utter += 1
    acc_score_all = float(accurate_utter)/num_utter
    print("accuracy with three tokens (action+object+location):" + str(acc_score_all))
    return classrep, acc_score,acc_score_all

def writeToFile(classrep, acc_score, acc_score_all,path):
    output = ''
    output += "accuracy score: " + str(acc_score)+"\n"
    output += "accuracy with three tokens (action+object+location): " + str(acc_score_all)+"\n"
    output += classrep + "\n"
    f = open(path+'/acc_score.txt','w+')
    f.write(output)
    print("Done!(Report in: {}".format(os.path.join(path,"acc_score.txt)")))

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.expdir
    dict_file = args.dict 
    #print(os.path.dirname(__file__))
    print("path in python ",path)
    num_utter, tokenid, rec_tokenid = read_test_result(path)
    # classification_report, acc_score(tokenid, rec_tokenid)
    classrep, acc_score, acc_score_all = acc_score(tokenid, rec_tokenid,num_utter,dict_file)
    writeToFile(classrep, acc_score, acc_score_all, args.expdir)
    #print('number of utterances:' + str(num_utter))


# print('number of length mismatch: ' + str(len_mismatch))
# print('number of content mismatch: ' + str(content_mismatch))
# print('ground truth len is 3 and pred is 4: ' + str(pred4))

