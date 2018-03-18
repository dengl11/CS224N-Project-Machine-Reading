""" prediction layer """

import sys 
from pred_basic import *
from pred_dense_softmax import * 
from pred_condition import * 

def get_prediction_layer(name, hidden_sz):
    """get a prediction layer class
    Args:
        name: 

    Return: 
    """
    print 'Using prediction layer %s' % name
    if name == "basic":
        return PredictionBasic()
    if name == "dense+softmax":
        return PredictionDenseSoftmax(hidden_sz)
    if name == "condition":
        return PredictionCondition(hidden_sz)
    else:
        sys.exit("No such Prediction Layer: {}!".format(name))
