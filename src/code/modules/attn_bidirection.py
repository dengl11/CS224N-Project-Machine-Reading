###########################################################
###############  BiDirectional Attention  #################
###########################################################

import sys, os
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from lib.util.logger import ColoredLogger 
from attn_basic import BasicAttn
from softmax import masked_softmax

logger = ColoredLogger("BiAttn")

class BiAttn(BasicAttn):

    """class for Bidirectional Attention"""

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        BasicAttn.__init__(self, keep_prob, key_vec_size, value_vec_size)
        self.scope = "BiAttn" 


    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Args:
            values:       [batch_sz, M, h]
            values_mask:  [batch_sz, M] 
            keys:         [batch_sz, N, h]
            
            (N = n_keys, M = n_values, h = hidden_size)

        Return: 
            attn_dist:    [batch_sz, N, num_values]
            output:       [batch_sz, N, output_sz]
        """
        h = self.key_vec_size
        M = values.shape[1]
        N = keys.shape[1] 
        assert(values.shape[-1] == h)

        with vs.variable_scope(self.scope): 
            # 3 similarity vectors : w = [w1; w2; w3]
            w = []
            for i in range(3): 
                v = tf.get_variable("sim_vec_{}".format(i+1),
                                     [h],
                                     tf.float32,
                                     tf.contrib.layers.xavier_initializer())
                w.append(v)

            ###### similarity matrix: [batch_sz, N, M] ###### 
            # kw2 = keys x w3: [batch_sz, N, h]
            _kw3 = tf.matmul(tf.reshape(keys, [-1, h]), tf.diag(w[2])) # [batch_sz*N, h]
            kw3  = tf.reshape(_kw3, [-1, N, h])

            # [batch_sz, N, M]
            S = tf.matmul(kw3, tf.transpose(values, [0, 2, 1]))

            # [N, batch_sz, M]
            S = tf.transpose(S, [1, 0, 2])  
            # values * w[1]: [batch_sz, M, h]
            S = S + tf.reduce_sum(values * w[1], 2) 
            # [M, batch_sz, N]
            S = tf.transpose(S, [2, 1, 0])  
            S = S + tf.reduce_sum(keys * w[0], 2)
            # [batch_sz, N, M]
            S = tf.transpose(S, [1, 2, 0])
            
            # ----------  key-to-value attention (C2Q) ----------
            # [N, batch_sz, M]
            _, alpha = masked_softmax(tf.transpose(S, [1, 0, 2]), values_mask, 2)
            # [batch_sz, N, M]
            alpha = tf.transpose(alpha, [1, 0, 2])
            # [batch_sz, N, h]
            k2v_attn = tf.matmul(alpha, values)
            
            # ----------  intermediate value-to-key attention (Q2C) ----------
            # [batch_sz, N]
            M = tf.reduce_max(S, 2)
            # [batch_sz, N]
            beta = tf.nn.softmax(M)

            c_prime = tf.transpose(keys, [2, 0, 1]) * beta  # [h, batch_sz, N]
            # [batch_sz, h]
            c_prime = tf.transpose(tf.reduce_sum(c_prime, 2), [1, 0])

            # ----------  final value-to-key attention (Q2C) ----------
            elems = [keys, k2v_attn, keys * k2v_attn, keys * tf.expand_dims(c_prime, 1)]
            # [batch_sz, N, 4 * h]
            v2k_attn = tf.concat(elems, 2)

            # Apply dropout
            v2k_attn = tf.nn.dropout(v2k_attn, self.keep_prob)

            return None, v2k_attn 
