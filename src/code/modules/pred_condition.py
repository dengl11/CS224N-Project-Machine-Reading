""" basic prediction layer """
from tensorflow.python.ops import variable_scope as vs
from softmax import *
import sys, os
from pred_basic import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from lib.util.logger import ColoredLogger 

logger = ColoredLogger("PredCond")

class PredictionCondition(PredictionBasic):

    """class for PredictionBasic. """

    def __init__(self, hidden_sz):
        self.scope = "pred_condition"
        self.hidden_sz = hidden_sz 
  
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
        cx_len = context_mask.shape[1]
        with vs.variable_scope(self.scope): 
            reps = tf.contrib.layers.fully_connected(reps,
                                                     num_outputs=self.hidden_sz) 
            logits_start, probdist_start = self._pred_start(reps, context_mask)
            end_reps = tf.concat([reps, tf.expand_dims(probdist_start, 2)], 2)
            # [batch_sz]: index of starting word
            # start_idx = tf.argmax(probdist_start, 1)
            # # [batch_sz, context_length]: 1 if valid for end word else 0
            # start_mask = 1 - tf.sequence_mask(start_idx, cx_len, dtype=tf.int32)
            # a position is valid for end work if both context mask and start mask are both 1
            # context_mask = context_mask * start_mask 

            logits_end,   probdist_end   = self._pred_end(end_reps, context_mask)
            return (logits_start, probdist_start, logits_end, probdist_end)

