""" basic prediction layer """
from tensorflow.python.ops import variable_scope as vs
from softmax import *

class PredictionBasic(object):

    """class for PredictionBasic. """

    def __init__(self):
        self.scope = "pred_basic"
  
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
        logits_start, probdist_start = self._pred_start(reps, context_mask)
        logits_end,   probdist_end   = self._pred_end(reps, context_mask)
        return (logits_start, probdist_start, logits_end, probdist_end)

           
    def _pred_start(self, reps, context_mask):
        """
        Args:
            reps: output representation 
                  [batch_sz, context_length, hidden_sz]
            context_mask:
                  [batch_sz, context_length]

        Return: 
        """
        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start,
        # both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            logits_start, probdist_start = softmax_layer_start.\
                            build_graph(reps, context_mask)
            return (logits_start, probdist_start)

    def _pred_end(self, reps, context_mask):
        """
        Args:
            reps: output representation 
                  [batch_sz, context_length, hidden_sz]
            context_mask:
                  [batch_sz, context_length]

        Return: 
        """
        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, 
        # both of which have shape [batch_size, context_len]
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            logits_end, probdist_end = softmax_layer_end.\
                            build_graph(reps, context_mask)
            return (logits_end, probdist_end)
