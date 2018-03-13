###########################################################
###############       Basic Attention     #################
###########################################################

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from softmax import masked_softmax

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.scope = "BasicAttn" 


    def get_key2value_attn(self, values, values_mask, keys, keys_mask):
        """
        Return: (attn_dist, attn_output)
        """
        with vs.variable_scope(self.scope): 
            # Calculate attention distribution
            # [batch_size, value_vec_size, num_values]
            values_t         = tf.transpose(values, perm=[0, 2, 1]) 
            # [batch_size, num_keys, num_values]
            attn_logits      = tf.matmul(keys, values_t) 
            # [batch_size, 1, num_values]
            attn_logits_mask = tf.expand_dims(values_mask, 1) 
            # [batch_size, num_keys, num_values]. take softmax over values
            _, attn_dist     = masked_softmax(attn_logits, attn_logits_mask, 2) 

            # Use attention distribution to take weighted sum of values
            # [batch_size, num_keys, value_vec_size]
            output = tf.matmul(attn_dist, values) 

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


    def build_graph(self, values, values_mask, keys, keys_mask=None):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape [batch_size, num_values, value_vec_size].
          values_mask: Tensor shape [batch_size, num_values]
            1s where there's real input, 0s where there's padding
          keys: Tensor shape [batch_size, num_keys, value_vec_size]

        Outputs:
          attn_dist: Tensor shape [batch_size, num_keys, num_values]
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape [batch_size, num_keys, hidden_size]
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        return self.get_key2value_attn(values, values_mask, keys, keys_mask)

