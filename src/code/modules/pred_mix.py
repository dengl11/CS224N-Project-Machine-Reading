""" basic prediction layer """
from tensorflow.python.ops import variable_scope as vs
from softmax import *
import sys, os
from pred_basic import *
from encoder import RNNEncoder 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from lib.util.logger import ColoredLogger 

logger = ColoredLogger("PredMix")

class PredictionMix(PredictionBasic):

    """class for PredictionBasic. """

    def __init__(self, hidden_sz, is_training):
        self.scope = "pred_mix"
        self.hidden_sz = hidden_sz 
        self.is_training = is_training 
  
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
            start_reps = tf.contrib.layers.fully_connected(reps,
                                                     num_outputs=self.hidden_sz)
            logits_start, probdist_start = self._pred_start(start_reps, context_mask)

            end_reps    = tf.concat([reps, tf.expand_dims(probdist_start, 2)], 2)
            end_encoder = RNNEncoder(self.hidden_sz, 1, "lstm", "end_encoder")
            end_reps    = end_encoder.build_graph(end_reps, context_mask)
            logits_end,   probdist_end   = self._pred_end(end_reps, context_mask)
            
            if not self.is_training:
                # [batch_sz]: index of starting word
                start_idx = tf.argmax(probdist_start, 1)
                # # [batch_sz, context_length]: 1 if valid for end word else 0.001
                start_mask = 1 - 0.999 * tf.cast(tf.sequence_mask(start_idx, cx_len, dtype=tf.int32),
                                                 tf.float32) 
                # a position is valid for end work if both context mask and start mask are both 1 
                logits_end   = logits_end * start_mask
                probdist_end = probdist_end * start_mask
            return (logits_start, probdist_start, logits_end, probdist_end)

