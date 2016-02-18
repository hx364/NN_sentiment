__author__ = 'Hao'
import numpy as np

def expand_label(y):
    """
    transform an 1-d y array of (length, ) to shape(length, 2)
    :param y:
    :return:
    """
    return np.array([np.ones_like(y)-y, y]).T

def pad_2Dsequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    "modify the keras.preprocessing.sequence to make it padding on doc level"
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    nb_dims = sequences[0].shape[1]

    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, nb_dims))*value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx,:len(trunc)] = trunc
        elif padding == 'pre':
            x[idx,-len(trunc):] = trunc
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

    return x
