import sys
import numpy as np

# KEY_WORDS = set(["how", "when", "where", "who", "what", "why"])

KEY_WORD_IDS = set([199, 63, 113, 40, 104, 740])
feature_sz = 2

def get_question_features(word2id, cx_ids, qn_ids, qn_mask):
    """
    Args:
        qn_ids: [batch_sz, qn_length]
        cx_ids: [batch_sz, cx_length]
        qn_mask: [batch_sz, cx_length]

    Return: 
        [batch_sz, qn_length, feature_sz]
    """
    # print([word2id[x] for x in KEY_WORDS])
    # sys.exit(0)
    batch_sz, qn_length = qn_ids.shape 
    features = np.zeros([batch_sz, qn_length, feature_sz])
    for bi, qn in enumerate(qn_ids):
        cx_set = set(list(cx_ids[bi]))
        for wi, w in enumerate(qn):
            if qn_mask[bi][wi] == 0: continue
            # feature1: key word ID 
            if w in KEY_WORD_IDS:
                features[bi][wi][0] = 1
            # feature2: appear in question 
            if w in cx_set:
                features[bi, wi][1] = 1
    return features 

def get_context_features(word2id, cx_ids, qn_ids, cx_mask):
    """
    Args:
        qn_ids: [batch_sz, qn_length]
        cx_ids: [batch_sz, cx_length]
        qn_mask: [batch_sz, cx_length]

    Return: 
        [batch_sz, qn_length, feature_sz]
    """
    batch_sz, cx_length = cx_ids.shape 
    features = np.zeros([batch_sz, cx_length, feature_sz])
    for bi, cx in enumerate(cx_ids):
        qn_set = set(list(qn_ids[bi]))
        for wi, w in enumerate(cx):
            if cx_mask[bi][wi] == 0: continue
            # feature1: key word ID 
            if w in KEY_WORD_IDS:
                features[bi][wi][0] = 1
            # feature2: appear in questions  
            if w in qn_set:
                features[bi][wi][1] = 1
    return features 

