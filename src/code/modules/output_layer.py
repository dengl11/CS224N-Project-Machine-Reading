""" output layer """

import sys 
from output_basic import OutputBasic
from output_lstm import *
from output_double_lstm import *
from output_double_lstm_dense import *

def get_output_layer(name, output_sz, keep_prob):
    """get a output layer class
    Args:
        name: 

    Return: 
    """
    if name == "basic":
        return OutputBasic(output_sz, keep_prob)
    elif name == "lstm":
        return OutputLSTM(output_sz, keep_prob)
    elif name == "double_lstm":
        return OutputDoubleLSTMDense(output_sz, keep_prob)
    elif name == "double_lstm_dense":
        return OutputDoubleLSTMDense(output_sz, keep_prob)
    else:
        sys.exit("No such Output Layer!")
