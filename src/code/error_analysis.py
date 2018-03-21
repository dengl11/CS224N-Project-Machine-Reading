from __future__ import absolute_import
from __future__ import division
import json, sys
from evaluate import *
import numpy as np 

DEV_PRED = "./ensumble_prediction.json"
# DEV_PRED = "./predictions.json"
DEV_DATA = "../data/dev-v1.1.json"
DEV_QN = "../data/dev.question"

# {ID: answer}
predictions = json.load(open(DEV_PRED))
dev_truth   = json.load(open(DEV_DATA))['data']

def eval_all():
    """
    Args:
        mode: 

    Return: 
    """
    output = "./error_analysis/eval_all.json"
    f1, em = [], []
    for passage in dev_truth:
        for p in passage['paragraphs']:
            qas = p['qas']
            for qa in qas:
                qn = qa['question']
                aws = [m['text'] for m in qa['answers']]
                id = qa['id']
                pred = predictions[id]
                curr_f1 = max(f1_score(pred, a) for a in aws)
                curr_em = max(exact_match_score(pred, a) for a in aws)
                f1.append(curr_f1)
                em.append(int(curr_em))
    with open(output, "w") as f:
        json.dump({"f1": f1, "em": em}, f)

def eval_by_len(mode, qn_an):
    """
    Args:
        mode: 

    Return: 
    """
    output = "./error_analysis/{}_by_{}.json".format(mode, qn_an)
    eval_fn = f1_score if mode == "f1" else exact_match_score
    dic = dict() # {qn_len: [n, acc_f1]}
    for passage in dev_truth:
        for p in passage['paragraphs']:
            qas = p['qas']
            for qa in qas:
                qn = qa['question']
                aws = [m['text'] for m in qa['answers']]
                id = qa['id']
                pred = predictions[id]
                val_value = max(eval_fn(pred, a) for a in aws)
                n = len(qn) if qn_an == "qn" else (np.mean([len(x) for x in aws]))
                n = int(n)
                if n in dic:
                    dic[n][0] += 1
                    dic[n][1] += val_value
                else:
                    dic[n] = [1, val_value]
    ans = dict()
    for k, v in dic.items():
        ans[k] = v[1]/v[0]
    with open(output, "w") as f:
        json.dump(ans, f)


eval_all()
eval_by_len("f1", "qn")
eval_by_len("f1", "an")
eval_by_len("em", "qn")
eval_by_len("em", "an")
