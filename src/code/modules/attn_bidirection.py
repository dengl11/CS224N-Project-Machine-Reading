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
            values:       [batch_sz, M, hidden_sz]
            values_mask:  [batch_sz, M] 
            keys:         [batch_sz, N, hidden_sz]
            
            (N = n_keys, M = n_values)

        Return: 
            attn_dist:    [batch_sz, N, num_values]
            output:       [batch_sz, N, output_sz]
        """
        hidden_sz          = self.key_vec_size
        batch_sz, M, _     = values.shape
        N                  = keys.shape[1] 
        assert(values.shape[-1] == hidden_sz)

        with vs.variable_scope(self.scope): 
            # [batch_sz, N, M]
            values_mask_exp = tf.tile(tf.expand_dims(values_mask, 1), [1, N, 1])
            # 3 similarity vectors : w = [w1; w2; w3]
            w = []
            for i in range(3): 
                v = tf.get_variable("sim_vec_{}".format(i+1),
                                             [hidden_sz],
                                             tf.float32,
                                             tf.contrib.layers.xavier_initializer())
                w.append(v)

            ###### similarity matrix: [batch_sz, N, M] ###### 
            # keys x w3: [batch_sz, N, hidden_sz]
            _kw3 = tf.matmul(tf.reshape(keys, [-1, hidden_sz]), tf.diag(w[2]))
            kw3  = tf.reshape(_kw3, [-1, N, hidden_sz])

            # [batch_sz, N, M]
            S = tf.matmul(kw3, tf.transpose(values, [0, 2, 1]))

            # [N, batch_sz, M]
            S = tf.transpose(S, [1, 0, 2])  
            S = S + tf.reduce_sum(values * w[1], 2) 
            # [M, batch_sz, N]
            S = tf.transpose(S, [2, 1, 0])  
            S = S + tf.reduce_sum(keys * w[0], 2)
            # [batch_sz, N, M]
            S = tf.transpose(S, [1, 2, 0])
            
            # ----------  key-to-value attention (C2Q) ----------
            # [batch_sz, N, M]
            _, alpha = masked_softmax(S, values_mask_exp, 2)
            # [batch_sz, N, hidden_sz]
            k2v_attn = tf.matmul(alpha, values)
            
            # ----------  intermediate value-to-key attention (Q2C) ----------
            # [batch_sz, N]
            M = tf.reduce_max(S, 2)
            # [batch_sz, N]
            beta = tf.nn.softmax(M)

            keys = tf.transpose(keys, [2, 0, 1]) # [hidden_sz, batch_sz, N]
            c_prime = keys * beta  # [hidden_sz, batch_sz, N]
            # [batch_sz, hidden_sz]
            c_prime = tf.transpose(tf.reduce_sum(c_prime, 2), [1, 0])
            # [batch_sz, hidden_sz, N]
            keys = tf.transpose(keys, [1, 2, 0]) # [batch_sz, N, hidden_sz]

            # ----------  final value-to-key attention (Q2C) ----------
            elems = [keys, k2v_attn, keys * k2v_attn, keys * tf.expand_dims(c_prime, 1)]
            # [batch_sz, N, 4 * hidden_sz]
            v2k_attn = tf.concat(elems, 2)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return None, v2k_attn 
