############################################################
###############  Toolkit for Numeric Operations  ###########
############################################################
import numpy as np
from collections import Counter
import statistics 

def mode(arr):
    """return mode of the array
    Args:
        arr: 

    Return: 
    """
    try:
        return statistics.mode(arr)
    except:
        c = Counter(arr)
        return c.most_common(1)[0][0]

def max_index(arr):
    """return the index item that have the max value 
    Args:
        arr: 

    Return: 
    """
    return max((v, k) for (k, v) in enumerate(arr))[1]
