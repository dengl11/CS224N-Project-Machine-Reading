"""This file contains some Encoder components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(rnn_cell_fw,
                                          input_keep_prob=self.keep_prob)
        rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(rnn_cell_bw,
                                          input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape [batch_size, seq_len, input_size]
          masks: Tensor shape [batch_size, seq_len]
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure
            tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            # [batch_size]
            input_lens = tf.reduce_sum(masks, reduction_indices=1) 

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(\
                                    self.rnn_cell_fw, self.rnn_cell_bw,
                                    inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out
