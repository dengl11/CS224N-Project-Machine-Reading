###########################################################
###############  BiDirectional Attention  #################
###########################################################

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from attn_basic import BasicAttn


class BidirectionalAttn(BasicAttn):
    def build_graph(self, questions, questions_mask, context):
        with vs.variable_scope("BidirectionalAttn"):
            # key_vec_size = 2h; sim_weight = 6h.
            sim_weight = tf.get_variable(shape=[3 * self.key_vec_size, ], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sim_weight')
            batch_size, question_length, question_state_size = questions.shape
            batch_size, context_length, context_state_size = context.shape

            original_context = context
            original_questions = questions

            # Calculate similarity matrix.
            context = tf.tile(tf.expand_dims(context, 2), (1, 1, question_length, 1)) # batch_size * context_length * question_length * 2h
            questions = tf.tile(tf.expand_dims(questions, 1), (1, context_length, 1, 1)) # batch_size * context_length * question_length * 2h
            context_questions_element_product = tf.multiply(context, questions)
            similarity_matrix = tf.reduce_sum(tf.multiply(sim_weight, tf.concat([context, questions, context_questions_element_product], 3)), 3)   # batch_size * context_length * question_length

            # Perform Context-to-Question attention.
            new_questions_mask = tf.tile(tf.expand_dims(questions_mask, 1), (1, context_length, 1))
            _, c_to_q_attention_dist = masked_softmax(similarity_matrix, new_questions_mask, 2)   # batch_size * context_length * question_length
            alpha = tf.squeeze(tf.matmul(tf.transpose(questions, (0, 1, 3, 2)), tf.expand_dims(c_to_q_attention_dist, 3)), 3)   # batch size * context_length * 2h

            # Perform Question-to-Context attention.
            beta = tf.nn.softmax(tf.reduce_max(similarity_matrix, 2))   # batch_size * context_length
            c_prime = tf.squeeze(tf.matmul(tf.transpose(original_context, (0, 2, 1)), tf.expand_dims(beta, 2)), 2) # batch size * 2h

            # Lastly.
            third_term = tf.multiply(original_context, alpha)
            fourth_term = tf.multiply(original_context, tf.tile(tf.expand_dims(c_prime, 1), (1, context_length, 1)))
            return None, tf.concat([original_context, alpha, third_term, fourth_term], 2)

