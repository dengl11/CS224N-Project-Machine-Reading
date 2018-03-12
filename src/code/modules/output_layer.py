""" output layer """

import sys 
from output_basic import *

def get_output_layer(name, output_sz):
    """get a output layer class
    Args:
        name: 

    Return: 
    """
    if name == "basic":
        return OutputRep("output_basic", output_sz)
    else:
        sys.exit("No such Output Layer!")
