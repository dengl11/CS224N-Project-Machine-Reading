"""
analyze idf of vocabulary of the squad dataset
"""

from __future__ import division
from os import path 
import sys
sys.path.append(path.join(path.dirname(path.abspath(__file__)), "../"))
from collections import Counter
from lib.util.sys_ops import *
import numpy as np 
from lib.util.logger import ColoredLogger

_my_path = get_fpath(__file__)

POSTFIX = ["context"]
idf_save_path = path.join(_my_path, "../../data/context_idf.txt")

logger = ColoredLogger("cx_idf")
n_docs = 0


def transform_to_idf(doc_counter):
    """transform doc counts to idf
    Args:
        doc_counter: 

    Return: 
    """
    idf = dict()
    for w, c in doc_counter.items():
        idf[w] = np.log(n_docs/c)
    return idf 
    

def get_doc_counter():
    docs = Counter() # {term: # of docs that contain the term}
    fpath = path.realpath(path.join(_my_path, "../../data/train.context"))
    global n_docs
    with open(fpath, "r") as f:
        for line in f:
            n_docs += 1
            for w in line.split():
                docs[w] += 1
    return docs

def save_to_file(idf):
    """
    Args:
        idf: 

    Return: 
    """
    with open(idf_save_path, "w") as f:
        for w, idf in idf.items():
            print("{}\t{}".format(w, idf), file=f)
    

def main():
    doc_counger = get_doc_counter()
    idf = transform_to_idf(doc_counger)
    save_to_file(idf)

if __name__ == "__main__":
    main()
    
