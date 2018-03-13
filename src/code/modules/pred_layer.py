""" prediction layer """

import sys 
from pred_basic import *

def get_prediction_layer(name):
    """get a prediction layer class
    Args:
        name: 

    Return: 
    """
    if name == "basic":
        return PredictionBasic()
    else:
        sys.exit("No such Prediction Layer: {}!".format(name))
