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

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        BasicAttn.__init__(self, keep_prob, key_vec_size, value_vec_size)
        self.scope = "SelfAttn"
        self.encoder = RNNEncoder(key_vec_size, keep_prob, "gru")
        self.v_size = 40

    def build_graph(self, questions_hidden, questions_mask, context_hidden, context_mask):
        """
        Args:
            questions_hidden:[batch_sz, M, h]
            questions_mask:  [batch_sz, M]
            context_hidden:  [batch_sz, N, h]
            context_mask:    [batch_sz, N]

        Return:
            attn_dist:    [batch_sz, N, 2h]
            output:       _
        """
        h = self.key_vec_size
        M = questions_mask.shape[1]
        N = context_mask.shape[1]

        # v: [batch_size, context_len, h]
        _, v = super(SelfAttn, self).build_graph(questions_hidden, questions_mask, context_hidden, context_mask)

        with vs.variable_scope(self.scope):
            w_1 = tf.get_variable('W_1',
                                  [self.v_size, h],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
            w_2 = tf.get_variable('W_2',
                                  [self.v_size, h],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer())
            v_weight = tf.get_variable('v',
                                [self.v_size, ],
                                tf.float32,
                                tf.contrib.layers.xavier_initializer())

            ###### W_1 * v_j & W_2 * v_i & their sum ######
            # batch_size * self.v_size * context_length
            w1_vj_product = tf.matmul(tf.tile(tf.expand_dims(w_1, 0),
                                              [tf.shape(v)[0], 1, 1]),
                                      tf.transpose(v, [0, 2, 1]))
            # batch_size * self.v_size * context_length
            w2_vi_product = tf.matmul(tf.tile(tf.expand_dims(w_2, 0),
                                              [tf.shape(v)[0], 1, 1]),
                                      tf.transpose(v, [0, 2, 1]))

            # batch_size * context_length * self.v_size
            w1_vj_product = tf.transpose(w1_vj_product, [0, 2, 1])
            w2_vi_product = tf.transpose(w2_vi_product, [0, 2, 1])


            # batch_size * context_length (j)* context_length (i) * self.v_size
            w1vj_w2vi_sum = tf.add(
              tf.tile(tf.expand_dims(w1_vj_product, 2), [1, 1, N, 1]),
              tf.tile(tf.expand_dims(w2_vi_product, 1), [1, N, 1, 1])
            )



            ###### e_ji ######
            # batch_size * context_length (j)* context_length (i)
            e_ji = tf.reduce_sum(tf.multiply(tf.tanh(w1vj_w2vi_sum), v_weight), 3)



            #### alpha_i ####
            # batch_size * context_length * context_length
            new_context_mask = tf.tile(tf.expand_dims(context_mask, 1), [1, N, 1])
            _, alpha_i = masked_softmax(e_ji, new_context_mask, 2)
            # batch size * context_length * h
            alpha_i = tf.matmul(alpha_i, context_hidden)



            #### construct new h1, h2, h3, h4.... hN ####
            bidirectional_gru_input = tf.concat([v, alpha_i], 2)
            attn = self.encoder.build_graph(bidirectional_gru_input, context_mask)

            return None, attn
