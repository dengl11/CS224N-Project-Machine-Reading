""" prediction layer """

import sys 
from pred_basic import *

def get_prediction_layer(name, hidden_sz):
    """get a prediction layer class
    Args:
        name: 

    Return: 
    """
    if name == "basic":
        return PredictionBasic()
    if name == "dense+softmax":
        return PredictionDenseSoftmax(hidden_sz)
    else:
        sys.exit("No such Prediction Layer: {}!".format(name))
