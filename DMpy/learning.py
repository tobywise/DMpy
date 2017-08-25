import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse

"""
Defining models

Outcome must be first argument, followed by dynamic inputs, and lastly static parameters

Should return a tuple of containing value followed by any other dynamic parameters, followed by other
dynamic parameters that are used at the next step (in same order as function arguments), followed by other
dynamic estimates that aren't reused (e.g. pe)

"""

# TODO would be better for all functions to return a consistent number of outputs, could wrap to achieve this

def rescorla_wagner(o, v, alpha):
    """
    o: outcome
    v: prior value
    alpha: learning rate
    """
    pe = o - v
    value = v + alpha * pe

    return (value, pe)


def metalearning_pe(o, v, c, alpha0, c_alpha, m):

    """
    From Vinckier et al., 2016, Molecular Psychiatry

    C = confidence parameter, updated based on prediction error magnitude

    """

    pe = o - v

    confidence = c + c_alpha * ((2 - np.abs(pe)) / 2 - c)

    c_m_alpha = (alpha0 + m * confidence) / (1 + m * confidence)  # confirmatory alpha
    d_m_alpha = alpha0 / (1 + m * confidence)  # same for non-confirmatory outcomes

    alpha_m = T.switch(T.eq(v.round(), o),  # switch = theano equivalent of if/else statement
                       c_m_alpha,
                       d_m_alpha)

    value = v + alpha_m * pe

    return (value, confidence, c_m_alpha, d_m_alpha, pe)


def metalearning_pe_2(o, v, c, alpha0, c_alpha, m):

    """
    From Vinckier et al., 2016, Molecular Psychiatry

    C = confidence parameter, updated based on prediction error magnitude

    """

    pe = o - v

    confidence = c + c_alpha * ((2 - np.abs(np.tanh(pe * 5))) / 2 - c)

    c_m_alpha = (alpha0 + m * confidence) / (1 + m * confidence)  # confirmatory alpha
    d_m_alpha = alpha0 / (1 + m * confidence)  # same for non-confirmatory outcomes

    alpha_m = T.switch(T.eq(v.round(), o),  # switch = theano equivalent of if/else statement
                       c_m_alpha,
                       d_m_alpha)

    value = v + alpha_m * pe

    return (value, confidence, c_m_alpha, d_m_alpha, pe)


def metalearning_pe_2lr_2(o, v, c, alpha0, c_alpha_p, c_alpha_n, m):

    """
    Adapted from Vinckier et al., 2016, Molecular Psychiatry

    C = confidence parameter, updated based on prediction error magnitude

    c_alpha_p = confidence learning rate for increases in confidence
    c_alpha_n = confidence learning rate for decreases in confidence

    """

    pe = o - v

    # c_update = ((2 - np.abs(np.tanh(pe * 5))) / 2 - c)
    c_update = ((1 - np.tanh(np.abs(pe*2))) / 1 - c)

    confidence = T.switch(T.gt(c_update, 0),
                          c + c_alpha_p * c_update,  # positive learning rate update
                          c + c_alpha_n * c_update)  # negative learning rate update

    c_m_alpha = (alpha0 + m * confidence) / (1 + m * confidence)  # confirmatory alpha
    d_m_alpha = alpha0 / (1 + m * confidence)  # same for non-confirmatory outcomes

    alpha_m = T.switch(T.eq(v.round(), o),  # switch = theano equivalent of if/else statement
                       c_m_alpha,
                       d_m_alpha)

    value = v + alpha_m * pe

    return (value, confidence, c_m_alpha, d_m_alpha, pe, c_update)


def metalearning_pe_2lr(o, v, c, alpha0, c_alpha_p, c_alpha_n, m):

    """
    Adapted from Vinckier et al., 2016, Molecular Psychiatry

    C = confidence parameter, updated based on prediction error magnitude

    c_alpha_p = confidence learning rate for increases in confidence
    c_alpha_n = confidence learning rate for decreases in confidence

    """

    pe = o - v

    # c_update = ((2 - np.abs(np.tanh(pe * 5))) / 2 - c)
    c_update = ((2 - np.abs(pe)) / 2 - c)

    confidence = T.switch(T.gt(c_update, 0),
                          c + c_alpha_p * c_update,  # positive learning rate update
                          c + c_alpha_n * c_update)  # negative learning rate update

    c_m_alpha = (alpha0 + m * confidence) / (1 + m * confidence)  # confirmatory alpha
    d_m_alpha = alpha0 / (1 + m * confidence)  # same for non-confirmatory outcomes

    alpha_m = T.switch(T.eq(v.round(), o),  # switch = theano equivalent of if/else statement
                       c_m_alpha,
                       d_m_alpha)

    value = v + alpha_m * pe

    return (value, confidence, c_m_alpha, d_m_alpha, pe)



def sk1(o, v, beta, h, k, mu, rhat):
    """
    Sutton (1992)
    """

    pe = o - v  # prediction error

    beta = beta + mu * pe * h  # update beta

    k = T.exp(beta) / (rhat + T.exp(beta)) # update learning rate

    h = (h + k * pe) * T.maximum(0, 1 - k) # update h

    value = v + k * pe

    return (value, beta, h, k, pe)


def hgf_binary(o, muhat1, mu2, mu3, pi2, pi3, rho2, rho3, ka2, om2, om3):

    """
    Mathys et al., 2011
    """

    # 2nd level prediction
    t = 1

    muhat2 = mu2 + t * rho2

    ###########
    # 1st level
    ###########

    # prediction
    muhat1 = T.nnet.sigmoid(muhat2)

    # precision of prediction
    pihat1 = 1. / (muhat1 * (1-muhat1))

    # updates
    pi1 = np.inf
    mu1 = o

    # prediction error
    da1 = mu1 - muhat1

    ##############################
    # 2nd level - prediction above
    ##############################

    # precision of prediction
    pihat2 = 1. / (1. / pi2 + np.exp(ka2 * mu3 + om2))

    # updates
    pi2 = pihat2 + 1./pihat1
    mu2 = muhat2 + 1./pi2 * da1

    # volatility prediction error
    da2 = (1./pi2  + (mu2 - muhat2) ** 2) * pihat2 -1

    ###########
    # 3rd level prediction
    ###########

    # prediction
    muhat3 = mu3 + t * rho3

    # precision of prediction
    pihat3 = 1. / (1. / pi3 + t * om3)

    # weighting factor
    v3 = t * om3
    v2 = t * np.exp(ka2 * mu3 + om2)
    w2 = v2 * pihat2

    # updates
    pi3 = pihat3 + 1./2 * ka2 ** 2 * w2 * (w2 + (2 * w2 -1) * da2)

    pi3 = T.switch(T.lt(pi3, 0), pi3*0, pi3)  # error check

    mu3 = muhat3 + 1./2 * 1. / pi3 * ka2 * w2 * da2

    # volatility prediction error
    da3 = (1. / pi3 + (mu3 - muhat3) ** 2) * pihat3 -1

    return (muhat1, mu2, mu3, pi2, pi3, da1, da2, da3)










