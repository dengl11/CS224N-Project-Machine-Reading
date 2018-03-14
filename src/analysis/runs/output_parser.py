import re

def parse(raw_output_path):
    """parse the evaluation data from raw_output_path 
    Args:
        raw_output_path: 

    Return: 
        [iters, train_f1, train_em, dev_f1, dev_em]
    """
    iters = []
    train_f1, train_em, dev_f1, dev_em = [], [], [], []
    with open(raw_output_path, "r") as f:
        for line in f.readlines():
            if "F1" not in line: continue
            line = line.replace(",", "")
            mode = "train" if "Train" in line else "dev"
            curr_iter = int(line[line.find("Iter"):].split(" ")[1])
            if not iters or iters[-1] != curr_iter:
                iters.append(curr_iter)
            f1 = float(line[line.find("F1 score:"):].split(" ")[2])
            em = float(line[line.find("EM score:"):].split(" ")[-1])
            if mode == "train":
                train_f1.append(f1)
                train_em.append(em)
            else:
                dev_f1.append(f1)
                dev_em.append(em)
    assert(len(iters) == len(train_f1)) 
    assert(len(iters) == len(dev_f1)) 
    return [iters, train_f1, train_em, dev_f1, dev_em]
    # print("iters: {}".format(iters))
    # print("train_f1: {}".format(train_f1))
    # print("train_em: {}".format(train_em))
    # print("dev_f1: {}".format(dev_f1))
    # print("dev_em: {}".format(dev_em))


# parse("./A_basic_E_100_D_0.25_1/raw_output")
