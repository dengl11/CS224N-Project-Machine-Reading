###########################################################
###############        Co-Attention       #################
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

logger = ColoredLogger("CoAttn")

class CoAttn(BasicAttn):

    """class for CoAttention"""

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        BasicAttn.__init__(self, keep_prob, key_vec_size, value_vec_size)
        self.scope   = "CoAttn" 
        self.encoder = RNNEncoder(key_vec_size, keep_prob, "lstm")


    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Args:
            values:       [batch_sz, M, h]
            values_mask:  [batch_sz, M] 
            keys:         [batch_sz, N, h]
            keys_mask:    [batch_sz, N] 
            
            (N = n_keys, M = n_values, h = hidden_size)

        Return: 
            attn_dist:    [batch_sz, N, num_values]
            output:       [batch_sz, N, output_sz]
        """
        h = self.key_vec_size
        M = values.shape[1]
        N = keys.shape[1] 
        assert(values.shape[-1] == h)

        logger.error("values: {}".format(values.shape))
        with vs.variable_scope(self.scope): 
            # weight matrix: [h, h]
            W = tf.get_variable("W",
                                [h, h],
                                tf.float32,
                                tf.contrib.layers.xavier_initializer())
            # bias: [h]
            b = tf.get_variable("b",
                                [h],
                                tf.float32,
                                tf.zeros_initializer())
            # sentinel vectors for keys and values
            # k0, v0 = [tf.get_variable(name, [h, 1], tf.float32,
                                # tf.zeros_initializer()) for name in ("k0", "v0")]
            # sen_mat = tf.matmul(v0, tf.transpose(k0, [1, 0]))
            # logger.error("sen_mat: {}".format(sen_mat.shape))
            # [batch_sz * M, h]
            q_prime = tf.nn.tanh(tf.matmul(tf.reshape(values, [-1, h]), W) + b)
            # [batch_sz, M, h]
            q_prime = tf.reshape(q_prime, [-1, M, h])

            # affinity matrix: L = [batch_sz, N, M]
            # logger.error("values: {}".format(values.shape))
            # logger.error("tf.matmul(keys, tf.transpose(values, [0, 2, 1])): {}".format((tf.matmul(keys, tf.transpose(values, [0, 2, 1]))).shape))
            L = tf.matmul(keys, tf.transpose(q_prime, [0, 2, 1])) 
            logger.error("L: {}".format(L.shape))

            ############ C2Q ############
            # [batch_size, 1, M]
            values_mask = tf.expand_dims(values_mask, 1)
            # [batch_size, 1, N]
            keys_mask = tf.expand_dims(keys_mask, 1)

            # softmax for L over values: [batch_sz, N, M]
            _, alpha = masked_softmax(L, values_mask, 2)

            logger.error("alpha: {}".format(alpha.shape))
            # [batch_sz, N, h]
            k2v      = tf.matmul(alpha, values)
            logger.error("k2v: {}".format(k2v.shape))

            ############ Q2C ############
            _, beta = masked_softmax(tf.transpose(L, [0, 2, 1]), keys_mask, 2) # [batch_sz, M, N]
            # softmax for L over keys: [batch_sz, N, M]
            beta = tf.transpose(beta, [0, 2, 1])
            logger.error("beta: {}".format(beta.shape))
            # [batch_sz, M, h]
            v2k = tf.matmul(tf.transpose(beta, [0, 2, 1]), keys)
            logger.error("v2k: {}".format(v2k.shape))

            ############ Second Level Attn ############
            # [batch_sz, N, h]: alpha = [batch_sz, N, M], v2k = [batch_sz, M, h]
            s = tf.matmul(alpha, v2k)
            logger.error("s: {}".format(s.shape))

            # [batch_sz, N, 2 * h]
            lstm_inputs = tf.concat([s, k2v], 2)
            logger.error("lstm_inputs: {}".format(lstm_inputs.shape))
            # logger.error("keys mask: {}".format((tf.squeeze(keys_mask)).shape))
            attn = self.encoder.build_graph(lstm_inputs, tf.squeeze(keys_mask))
            logger.error("attn: {}".format(attn.shape))

            # Apply dropout
            attn = tf.nn.dropout(attn, self.keep_prob)

            return _, attn 
