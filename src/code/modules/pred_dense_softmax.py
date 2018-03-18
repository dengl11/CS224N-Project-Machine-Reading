""" basic prediction layer """
from tensorflow.python.ops import variable_scope as vs
from softmax import *
from pred_basic import *

class PredictionDenseSoftmax(PredictionBasic):

    """class for PredictionBasic. """

    def __init__(self, hidden_sz):
        self.scope = "pred_dense_softmax"
        self.hidden_sz = hidden_sz
        self.answer_length = 10
  
    def build_graph(self, reps, context_mask):
        """give the final prediction for start_pos and end_pos
        Args:
            reps: final output representation 
                  [batch_sz, context_length, hidden_sz]

            context_mask:
                  [batch_sz, context_length]
        Return: 
            (logits_start, probdist_start, logits_end, probdist_end)
            each of shape [batch_sz, context_length]
        """
        with vs.variable_scope(self.scope): 
            reps = tf.contrib.layers.fully_connected(reps,
                                                     num_outputs=self.hidden_sz) 
            logits_start, probdist_start = self._pred_start(reps, context_mask)
            logits_end,   probdist_end   = self._pred_end(reps, context_mask)
            # The value at each index is prob_end(index) + prob_end(index+1) ....
            end_prob_sum = tf.identity(probdist_end)
            # The value at each index is prob_start(index) + prob_start(index-1) ...
            start_prob_sum = tf.identity(probdist_start)

            for index in range(1, self.answer_length):
                padding_at_end = tf.constant([[0, 0], [0, index]])
                padding_at_start = tf.constant([[0, 0], [index, 0]])
                end_prob_sum = tf.add(end_prob_sum,
                                      tf.pad(probdist_end[:, index:],
                                             padding_at_end, "CONSTANT"))
                start_prob_sum = tf.add(end_prob_sum,
                                        tf.pad(probdist_start[:, :-index],
                                               padding_at_start, "CONSTANT"))
            probdist_start = tf.multiply(probdist_start, end_prob_sum)
            probdist_end = tf.multiply(probdist_end, start_prob_sum)

            probdist_start = tf.nn.l2_normalize(probdist_start, 1)
            probdist_end = tf.nn.l2_normalize(probdist_end, 1)

            return (logits_start, probdist_start, logits_end, probdist_end)

