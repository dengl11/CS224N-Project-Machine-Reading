###########################################################
###############       Dataframe Ops       #################
###########################################################
from pandas import DataFrame 

def mat2df(mat, index=None, columns=None):
    """return a dataframe from a np matrix 
    Args:
        mat: np matrix 

    Return: 
    """
    return DataFrame(mat, index=index, columns=columns)
    
