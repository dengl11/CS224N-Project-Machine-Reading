""" ensumble of model for prediction """
import os, sys, re 
from lib.util import sys_ops
from lib.util.logger import ColoredLogger
from lib.util.dot_dict import DotDict
from data_batcher import get_batch_generator
from scipy import stats
from evaluate import exact_match_score, f1_score
import tensorflow as tf
from qa_model import QAModel

logger = ColoredLogger("Ensumbler")
INT_ATTR = set(["context_len", "output_size", "embedding_size", "pred_hidden_sz", "batch_size", "hidden_size"])

FLT_ATTR = set(["dropout", "decay_rate", "learning_rate"])
  
class Ensumbler(object):


    def __init__(self, ensumble_path, config, id2word, word2id, emb_matrix, id2idf):
        self.id2word = id2word 
        self.models = self._init_models(config, ensumble_path, id2word, word2id, emb_matrix, id2idf) 
        self.id2idf = id2idf
        self.word2id = word2id
        self.batch_size = 100

    
    def _init_models(self, tf_config, ensumble_path, id2word, word2id, emb_matrix, id2idf):
        """get the predictions from an ensumble of models 
        Args:
            best_checkpoints: 

        Return: 
        """
        ensumble_path = os.path.abspath(ensumble_path)
        best_checkpoints = sys_ops.dirs_in_dir(ensumble_path)
        logger.info("Parsing config from {}...".format(best_checkpoints))
        self.sessions = []
        for dir_path in best_checkpoints:   
            FLAGS = parse_flags(dir_path)
            session = tf.Session(config=tf_config)
            # Initialize model
            qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix, id2idf, is_training=False)
	    checkpoint_path = os.path.join(dir_path, "best_checkpoint")
            qa_model.initialize_from_checkpoint(session, checkpoint_path, True)

            self.models.append(qa_model)
            self.sessions.append(session)


    def get_predictions(self, batch):
        """
        Return: 
        """
        starts, ends = [], []
        for m, session in zip(self.models, self.sessions):
            pred_start_pos, pred_end_pos = m.get_start_end_pos(session, batch)
            starts.append(pred_start_pos)
            ends.append(pred_end_pos)
        starts = stats.mode(np.array(starts)).mode 
        ends = stats.mode(np.array(ends)).mode 
        return (starts, ends)

    def check_f1_em(self, context_path, qn_path, ans_path, dataset, num_samples=100):
        f1_total = 0.
        em_total = 0.
        example_num = 0
        for batch in get_batch_generator(self.word2id, self.id2idf,
                                         context_path,
                                         qn_path, ans_path,
                                         self.FLAGS.batch_size,
                                         context_len=self.FLAGS.context_len,
                                         question_len=self.FLAGS.question_len,
                                         discard_long=False):

            pred_start_pos, pred_end_pos = self.get_predictions(batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in \
                    enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        return f1_total, em_total

def parse_flags(dir_path):
    """parse FLAGS from the Makefile in dir_path
    Args:
        dir_path: 

    Return: 
        config dictionary 
    """
    config_path = os.path.join(dir_path, "info.md")
    logger.info("Parsing config from {}...".format(config_path))
    config = dict()
    with open(config_path, "rb") as f:
        for line in f.readlines():
            line = line.lstrip()[:-2]
            if line[:2] != "--":
                continue
            try:
                k, v = re.split(" |=", line[2:])
                if k in INT_ATTR:
                    v = int(v)
                if k in FLT_ATTR:
                    v = float(v)
                config[k] = v
            except Exception as e:
                pass 
    if not "encoder" in config:
        config["encoder"] = "gru"
    if not "save_every" in config:
        config["save_every"] = 500
    if not "eval_every" in config:
        config["eval_every"] = 500
    if not "keep" in config:
        config["keep"] = 1
    if not "decay_rate" in config:
        config["decay_rate"] = 0.8
    if not "max_gradient_norm" in config:
        config["max_gradient_norm"] = 5.0
    if not "question_len" in config:
        config["question_len"] = 30

    logger.warn("Parse: {}".format(config))
    return DotDict(config )
