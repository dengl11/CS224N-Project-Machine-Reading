###########################################################
###############        Combined-Attention       #################
###########################################################

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from attn_bidirection import *
from attn_self import *

logger = ColoredLogger("CombinedAttn")


class CombinedAttn(BasicAttn):
    """Class for CombinedAttention"""

    def __init__(self, keep_prob, key_size, value_size):
      self.self_attn = SelfAttn(keep_prob, key_size, value_size)
      self.bi_attn = BiAttn(keep_prob, key_size, value_size)
      self.scope = 'CombinedAttn'
      self.keep_prob = keep_prob
      self.encoder = RNNEncoder(key_size, keep_prob, "gru", scope="combined_attn")

    def build_graph(self, values, values_mask, keys, keys_mask):
      # Concatenate the attn.
      _, self_attn_result = self.self_attn.build_graph(values, values_mask, keys, keys_mask)
      _, bi_attn_result = self.bi_attn.build_graph(values, values_mask, keys, keys_mask)
      concatenated_attn = tf.concat([self_attn_result, bi_attn_result], 2)

      # Encode the concatenated attn, to combine info from both attn and shrink size.
      attn = self.encoder.build_graph(concatenated_attn, keys_mask)

      # Apply dropout.
      attn = tf.nn.dropout(attn, self.keep_prob)
      return None, attn
