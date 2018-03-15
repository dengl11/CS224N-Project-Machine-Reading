"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove, get_idf
from official_eval_helper import get_json_data, generate_answers

from lib.util.logger import ColoredLogger
from lib.util.timer import Timer


logging.basicConfig(level=logging.INFO)

# relative path of the main directory
MAIN_DIR = os.path.relpath(
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
# relative path of data dir
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") 
# relative path of experiments dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") 

logger = ColoredLogger("main")
timer = Timer()

# High-level options
tf.app.flags.DEFINE_string("encoder", "gru",
                          "encoder after word embedding: gru or lstm")
tf.app.flags.DEFINE_string("attn_layer", "",
                          "atten layer after encoder")
tf.app.flags.DEFINE_string("output", "basic",
                          "output layer after attention before final softmax")
tf.app.flags.DEFINE_string("output_activation", "tanh",
                          "for output=lstm_act only, activation function after output")
tf.app.flags.DEFINE_string("pred_layer", "basic",
                          "prediction layer after output layer for final prediction")
tf.app.flags.DEFINE_string("idf_path", "../data/context_idf.txt",
                          "prediction layer after output layer for final prediction")

tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")

tf.app.flags.DEFINE_integer("char_vocab_sz", 95,
                            "vocabulary size of chars")
tf.app.flags.DEFINE_integer("char_encoding_sz", 10,
                            "0 if not use character encoding else some positive number")
tf.app.flags.DEFINE_string("mode", "train", 
                      "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "",
        "Unique name for your experiment.\
         This will create a directory by this name in the experiments/ directory,\
         which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0,
                        "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_integer("pred_hidden_sz", 100,
                            "hidden size for for prediction_dense_softmax")


# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15,
        "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 200,
                            "Size of the hidden states")
tf.app.flags.DEFINE_integer("output_size", 100,
                            "Size of the output")
tf.app.flags.DEFINE_integer("context_len", 600,
        "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30,
        "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100,
        "Size of the pretrained word vectors.\
        This needs to be one of the available GloVe dimensions: 50/100/200/300")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 200,
        "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500,
        "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500,
        "How many iterations to do per calculating loss/f1/em on dev set.\
        Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1,
        "How many checkpoints to keep. 0 indicates keep all \
        (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "",
        "Training directory to save the model parameters and other info.\
        Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "",
        "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, 
        "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", 
        "For official_eval mode, which directory to load the checkpoint fron.\
        You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", 
        "For official_eval mode, path to JSON input file. \
        You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json",
        "Output path for official_eval mode. Defaults to predictions.json")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
                     If False, initialize fresh model if no checkpoint is found.
    """
    logger.info("Looking for model at %s..." % train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            logger.warning("There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir)
            session.run(tf.global_variables_initializer())

            logger.error('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Print out Tensorflow version
    # print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    if not FLAGS.attn_layer and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --attn_layer or --train_dir")

    # Define train_dir
    if not FLAGS.experiment_name:
        FLAGS.experiment_name = "A_{}_E_{}_D_{}".format(FLAGS.attn_layer,
                                                        FLAGS.embedding_size,
                                                        FLAGS.dropout)

    checkptr_name = FLAGS.experiment_name + "/glove{}".format(FLAGS.embedding_size)
    FLAGS.train_dir = FLAGS.train_dir or\
                        os.path.join(EXPERIMENTS_DIR, checkptr_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or \
                       os.path.join(DEFAULT_DATA_DIR, 
                            "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    # Load embedding matrix and vocab mappings
    timer.start("glove_getter")
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)
    id2idf = get_idf(FLAGS.idf_path, word2id)
    logger.warn("Get glove embedding of size {} takes {:.2f} s".format(FLAGS.embedding_size, timer.stop("glove_getter")))

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = os.path.join(FLAGS.data_dir, "train.context")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
    dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

    # Initialize model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix, id2idf)

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train": 
        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess: 
            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.train(sess, train_context_path, train_qn_path,
                           train_ans_path, dev_qn_path, dev_context_path,
                           dev_ans_path)


    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            f1, em = qa_model.check_f1_em(sess, dev_context_path,
                                        dev_qn_path, dev_ans_path,
                                        "dev", num_samples=10,
                                        print_to_screen=True)
            logger.info("Dev: F1 = {0:.3}, EM = {0:.3}".format(f1, em))


    elif FLAGS.mode == "eval":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # train
            f1, em = qa_model.check_f1_em(sess, train_context_path,
                                        train_qn_path, train_ans_path,
                                        "train", num_samples=100000000000,
                                        print_to_screen=False)
            logger.error("Train: F1 = {0:.3}, EM = {0:.3}".format(f1, em))
            # dev
            f1, em = qa_model.check_f1_em(sess, dev_context_path,
                                        dev_qn_path, dev_ans_path,
                                        "dev", num_samples=10000000000,
                                        print_to_screen=False)
            logger.error("Dev:   F1 = {0:.3}, EM = {0:.3}".format(f1, em))

    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict = generate_answers(sess, qa_model, word2id,
                    qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print "Wrote predictions to %s" % FLAGS.json_out_path


    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
