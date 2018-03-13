###########################################################
###############  Final Output Representation     ##########
###########################################################

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from softmax import masked_softmax

class OutputBasic(object):

    """base class for output representation"""

    def __init__(self, output_sz, keep_prob):
        """
        Args:
        """
        self.output_sz = output_sz
        self.keep_prob = keep_prob
        self.scope = "output_basic"


    def build_graph(self, reps, context_mask):
        """
        Args:
             reps: [batch_sz, context_length, reps_sz]

        Return: 
             [batch_sz, context_length, output_sz]
        """
        # baseline: just a dense layer 
        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        # blended_reps_final is shape [batch_size, context_len, hidden_size]
        with vs.variable_scope(self.scope):
            return tf.contrib.layers.fully_connected(reps,
                                                     num_outputs=self.output_sz) 
