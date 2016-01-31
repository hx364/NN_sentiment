__author__ = 'Hao'
import numpy as np

def expand_label(y):
    """
    transform an 1-d y array of (length, ) to shape(length, 2)
    :param y:
    :return:
    """
    return np.array([np.ones_like(y)-y, y]).T
