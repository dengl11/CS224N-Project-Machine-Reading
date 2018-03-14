###########################################################
###############         Runs Analyzer            ##########
###########################################################

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lib.util.logger import ColoredLogger 
from lib.util.plotter import * 
from output_parser import *

logger = ColoredLogger("run_analyzer")
PARAM_FILE = "raw_output"

def plot_eval(data, dir_path):
    """plot train/dev curves for a run in dir_path 
    Args:
        data: [iters, train_f1, train_em, dev_f1, dev_em]

    Return: 
    """
    name = dir_path.split()[-1]
    iters = data.pop(0)
    for i, mode in enumerate(["F1", "EM"]):
        train, dev = data[i], data[2+i] 
        curve_plot(iters, train, show=False)
        curve_plot(iters, dev, 
                   new=False,
                   xlabel="Iters",
                   ylabel=mode, 
                   save_path = os.path.join(dir_path, "eval_{}.png".format(mode)),
                   show=False,
                   title="{} for {}".format(mode, name),
                   legend=["Train", "Dev"])
        logger.info("{} done!".format(mode))
    

def analyze_run(dir_path):
    """analyze a run in a dir_path
    Args:
        dir_path: 

    Return: 
    """
    file_path = os.path.join(dir_path, PARAM_FILE)
    if not os.path.isfile(file_path):
        return 
    logger.info("Analyzing runs in {} ...".format(dir_path))
    # [iters, train_f1, train_em, dev_f1, dev_em]
    data = parse(file_path)
    plot_eval(data, dir_path)


def main():
    for dir_path in os.listdir("."):
        if not os.path.isdir(dir_path): continue 
        analyze_run(dir_path)


if __name__ == "__main__":
    main()
