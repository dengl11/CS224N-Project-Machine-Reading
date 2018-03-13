""" attention layer """

import sys 
from .attn_basic import *
from .attn_bidirection import *
from .attn_co import *
from .attn_self import *

def get_attn_layer(name, keep_prob, context_size, question_size):
    """get an attention layer class
    Args:
        name: 

    Return: 
    """
    if name == 'basic':
        attn_layer = BasicAttn(keep_prob,
                               context_size,
                               question_size)
    elif name == 'bi_attn':
        attn_layer = BiAttn(keep_prob,
                            context_size,
                            question_size)
    elif name == 'co_attn':
        attn_layer = CoAttn(keep_prob,
                            context_size,
                            question_size)
    elif name == 'self_attn':
        attn_layer = SelfAttn(keep_prob,
                              context_size,
                              question_size)
    else:
        sys.exit("no such Attention Layer: {}!".format(name))

    return attn_layer 
