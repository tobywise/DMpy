import numpy as np

"""
Observation models should return (probability, other quantities... )
"""

def softmax(v, b):

    prob = (b * v).exp() / ((b * v).exp() + (b * (1 - v)).exp())

    return (prob, )


def softmax_ml(v, c, b, m):
    """
    From Vinckier et al (2016)
    """
    bb = b / (1 - m * c)
    return ((bb * v).exp() / ((bb * v).exp() + (bb * (1 - v)).exp()), [bb])


def softmax_ml2(v, c, b, m):
    """
    Modified metalearning softmax - tanh transform on C to prevent extreme b values
    """
    bb = b / (1 - m * np.tanh(c))
    return ((bb * v).exp() / ((bb * v).exp() + (bb * (1 - v)).exp()), [bb])

