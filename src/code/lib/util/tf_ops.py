############################################################
##############  Toolkit for Tensorflow Operations  #########
############################################################

import tensorflow as tf

def get_gpu_session():
    """
    return a tf session with gpu growth allowed
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def get_solvers_adam(learning_rate=1e-3, beta1=0.5):
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    return solver

def initWeight(n,d):
    """
    init a weight matrix W of size n, d
    """
    scale_factor = math.sqrt(float(6)/(n + d))
    return (np.random.rand(n,d)*2-1)*scale_factor
