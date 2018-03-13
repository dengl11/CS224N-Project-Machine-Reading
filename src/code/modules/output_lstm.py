###########################################################
###############  Final Output Representation     ##########
###########################################################

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from softmax import masked_softmax
from encoder import RNNEncoder

class OutputLSTM(object):

    """base class for output representation"""

    def __init__(self, output_sz, keep_prob):
        """
        Args:
        """
        self.output_sz = output_sz
        self.scope = "output_lstm"
        self.lstm_encoder = RNNEncoder(output_sz, keep_prob, "lstm")


    def build_graph(self, reps, context_mask):
        """
        Args:
             reps: [batch_sz, context_length, reps_sz]

        Return: 
             [batch_sz, context_length, output_sz]
        """
        with vs.variable_scope(self.scope):
            return self.lstm_encoder.build_graph(reps, context_mask)
