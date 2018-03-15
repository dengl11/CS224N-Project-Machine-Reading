###########################################################
###############  Double LSTM with Activation     ##########
###########################################################

import os, sys 
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from softmax import masked_softmax
from encoder import RNNEncoder
from output_double_lstm import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from lib.util.logger import ColoredLogger 

logger = ColoredLogger("OutputDoubleLSTMAct")

class OutputDoubleLSTMAct(object):

    """base class for output representation"""
    def __init__(self, output_sz, keep_prob, activation):
        """
        Args:
        """
        self.output_sz = output_sz
        self.activation = activation 
        self.scope = "double_lstm_{}".format(activation)
        self.keep_prob = keep_prob 
        self.lstm_encoder1 = RNNEncoder(output_sz, keep_prob, "lstm", "encoder1")
        self.lstm_encoder2 = RNNEncoder(output_sz, keep_prob, "gru", "encoder2")
        logger.error("Output Layer with Double LSTM and Activation {} created ...".format(activation))

    def build_graph(self, reps, context_mask):
        """
        Args:
             reps: [batch_sz, context_length, reps_sz]

        Return: 
             [batch_sz, context_length, output_sz]
        """
        with vs.variable_scope(self.scope):
            lstm_1_out = self.lstm_encoder1.build_graph(reps, context_mask)
            lstm_2_out = self.lstm_encoder2.build_graph(lstm_1_out, context_mask)
            if self.activation == "tanh":
                return tf.nn.tanh(lstm_2_out) 
            elif self.activation == "relu":
                return tf.nn.relu(lstm_2_out) 
            sys.exit(0, "No such activation: {}!".format(self.activation))
