from __future__ import absolute_import
from __future__ import division
import sys, os
import json, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from lib.util.logger import ColoredLogger 
from lib.util.plotter import * 
import seaborn as sns

def bucketize(x, y, n_bucket):
    """
    Args:
        x: 
        y: 
        n_bucket: 

    Return: 
    """
    interval = (x[-1]-x[0])/n_bucket
    new_x, new_y = [], []
    i = 0
    pre = 0
    curr_x = x[0]
    for i in range(len(x)):
        if x[i] > curr_x + interval or i == len(x) - 1:
            new_x.append(curr_x)
            new_y.append(np.mean(y[pre:i]))
            curr_x += interval
            pre = i 
    return new_x, new_y



def plot_eval_by_len(mode, qnan):
    """
    Args:
        mode: 
        qnan: 

    Return: 
    """
    json_input = "./{}_by_{}.json".format(mode, qnan)
    save_path = os.path.abspath("./{}_by_{}.png".format(mode, qnan))
    ylabel = "F1" if mode == "f1" else "EM"
    xlabel = "Question" if qnan == "qn" else "Answer"
    xlabel += " Length (Ground Truth)"
    title = "{} by {}".format(ylabel, xlabel)
    data = json.load(open(json_input))
    lens = sorted(int(x) for x in data.keys())
    vals = [data[str(k)] for k in lens]
    lens, vals = bucketize(lens, vals, 16)

    bar_plot(lens, vals, xlabel, ylabel, title, save_path = save_path)

def arr2dist(x, n_bucket):
    """
    Args:
        arr: 

    Return: 
    """
    x.sort()
    interval = (x[-1]-x[0])/n_bucket
    new_x, prob = [], []
    i = 0
    pre = 0
    curr_x = x[0]
    for i in range(len(x)):
        if x[i] > curr_x + interval or i == len(x) - 1:
            new_x.append(curr_x)
            prob.append((i-pre + 1)/len(x))
            curr_x += interval
            pre = i 
    return new_x, prob
    

def plot_eval_all():
    """
    return: 
    """
    json_input = "./eval_all.json"
    data = json.load(open(json_input))
    f1, em = data["f1"], data["em"]
    new = True 
    for mode, data in zip(["f1", "em"], [f1, em]):
        x, y = arr2dist(data, 400)
        mode = "f1" if mode == "f1" else "em"
        title = "Distribution of F1 & EM"
        if new:
            curve_plot(x, y, show=False)
            new = False 
        else:
            curve_plot(x, y, ylabel="Distribution", xlabel="F1/EM Value", title = title, show=False, save_path = os.path.abspath("eval_all.png"), new = False, legend=["F1", "EM"], color="pink")
    

def plot_len_math():
    json_input = "./ans_len_match.json"
    data = json.load(open(json_input))
    pred, gd = data["pred"], data["gd"]
    pred_x, pred_y = arr2dist(pred, 20)
    gd_x, gd_y = arr2dist(gd, 20)
    save_path = os.path.abspath("./ans_len_match.png")
    curve_plot(pred_x, pred_y, show=False, new=True)
    curve_plot(gd_x, gd_y, xlabel="Length", ylabel="Distribution", title="Answer Length Error Analysis", show=False, new=False, save_path = save_path, legend = ["Prediction", "Ground Truth"])


def plot_qn_type():
    json_input = "./eval_by_qn_type.json"
    data = json.load(open(json_input))
    xlabels = list(data.keys())
    f = [data[k][0] for k in xlabels]
    em = [data[k][1] for k in xlabels]
    hbar_plot(xlabels, f, "F1", "Question Type", "F1 by Question Type", save_path = os.path.abspath("./f1_by_qn_type.png"))
    hbar_plot(xlabels, em, "EM", "Question Type", "EM by Question Type", save_path = os.path.abspath("./em_by_qn_type.png"), color = "pink")


sns.set_style("darkgrid")
# plot_qn_type()
# plot_len_math() 
plot_eval_all() 
# plot_eval_by_len("f1", "qn") 
# plot_eval_by_len("f1", "an")
# plot_eval_by_len("em", "qn")
# plot_eval_by_len("em", "an")



