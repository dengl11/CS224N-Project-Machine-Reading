"""
analyze vocabulary of the squad dataset
"""
from os import path
import sys
sys.path.append(path.join(path.dirname(path.abspath(__file__)), "../"))
from itertools import product
from lib.util.sys_ops import *
from lib.util.logger import ColoredLogger

_my_path = get_fpath(__file__)

MODES   = ["train", "dev"]
POSTFIX = ["context", "question", "answer"]

logger = ColoredLogger("vocab analyzer")

def add_words_from_file(path, vocab):
    """add words from file to vocabulary
    Args:
        path: 
        vocab: {} set of words

    Return: 
    """
    logger.info("adding vocabulary from {}".format(path.split("/")[-1]))
    with open(path) as f:
        for line in f:
            vocab.update(line.split())
    logger.warn("current vocaculary size: {}\n".format(len(vocab)))
    

def get_squad_vocab():
    """get the vocabulary of squad train/dev dataset 
    Return: {} set of words
    """
    vocab = set()
    for mode, postfix in product(MODES, POSTFIX):
        fpath = path.realpath(
                path.join(_my_path, "../../data/{}.{}".format(mode, postfix)))
        add_words_from_file(fpath, vocab)
    print list(vocab)[:100]


if __name__ == "__main__":
    get_squad_vocab()
