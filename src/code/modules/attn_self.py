###########################################################
###############        Self-Attention       #################
###########################################################

import sys, os
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from lib.util.logger import ColoredLogger
from attn_basic import BasicAttn
from encoder import RNNEncoder
from softmax import masked_softmax

logger = ColoredLogger("SelfAttn")


class SelfAttn(BasicAttn):
    """class for SelfAttention"""

    def __init__(self, keep_prob, key_size, value_size):
        BasicAttn.__init__(self, keep_prob, key_size, value_size)
        self.scope = "SelfAttn"
        self.encoder = RNNEncoder(key_size, keep_prob, "gru")
        self.v_size = 35

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Args:
            values:       [batch_sz, M, h]
            values_mask:  [batch_sz, M]
            keys:         [batch_sz, N, h]
            keys_mask:     [batch_sz, N]

        Return:
            attn_dist:    [batch_sz, N, 2h]
            output:       _
        """
        h = self.key_vec_size
        M = values_mask.shape[1]
        N = keys_mask.shape[1]
        v = self.v_size 
        # convert keys to first level attention
        _, keys = super(SelfAttn, self).build_graph(values,
                                                 values_mask,
                                                 keys,
                                                 keys_mask)

        with vs.variable_scope(self.scope):
            W_1 = tf.get_variable('W_1',
                                  [h, v],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
            W_2 = tf.get_variable('W_2',
                                  [h, v],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
            v_weight = tf.get_variable('v',
                                [v, 1],
                                tf.float32,
                                tf.contrib.layers.xavier_initializer())

            ###### W_1 * v_j & W_2 * v_i & their sum ######
            keys = tf.reshape(keys, [-1, h])
            # [batch_sz, N, N, v] - v: self.v_size
            W1v = tf.tile(tf.expand_dims(\
                          tf.reshape(tf.matmul(keys, W_1), [-1, N, v]),\
                          2), [1, 1, N, 1])
            # [batch_sz, N, N, v]
            W2v = tf.tile(tf.expand_dims(\
                          tf.reshape(tf.matmul(keys, W_2), [-1, N, v]),\
                          2), [1, 1, N, 1])
            # restore keys to [batch_sz, N, h]
            keys = tf.reshape(keys, [-1, N, h])

            # [batch_sz, N, N, v]
            # each vector in W_mixed (i, j) is W1v_i + W2v_j 
            W_mixed = W1v + tf.transpose(W2v, [0, 2, 1, 3])
            # [batch_sz * N, N]
            E = tf.matmul(tf.reshape(W_mixed, [-1, v]), v_weight) 
            # [batch_sz, N, N]
            E = tf.reshape(E, [-1, N, N])
            # [N, batch_sz, N]
            _, alpha = masked_softmax(tf.transpose(E, [1, 0, 2]), keys_mask, 2)
            # [batch_sz, N, N]
            alpha = tf.transpose(alpha, [1, 0, 2])
            # [batch_sz, N, h]
            alpha = tf.matmul(alpha, keys) 

            #### Bi-RNN ####
            bidirectional_gru_input = tf.concat([keys, alpha], 2)
            attn = self.encoder.build_graph(bidirectional_gru_input, keys_mask)

            attn = tf.nn.dropout(attn, self.keep_prob)

            return None, attn
