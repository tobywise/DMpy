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

def rescorla_wagner(o, t, v, alpha):
    """
    o: outcome
    v: prior value
    alpha: learning rate
    """
    pe = o - v
    value = v + alpha * pe

    return (value, pe)


def metalearning_pe(o, t, v, c, alpha0, c_alpha, m):

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


def sk1(o, t, v, beta, h,  mu, rhat):

    """
    Sutton (1992)
    """

    pe = o - v  # prediction error

    beta = T.switch(T.eq(t, 0), T.log(rhat), beta)
    beta = beta + mu * pe * h  # update beta

    phatii = T.exp(beta)

    k = phatii / (rhat + phatii) # update learning rate

    h = (h + k * pe) * T.maximum(0, 1 - k) # update h

    value = v + k * pe

    return (value, beta, h, k, pe, phatii)


def hgf_binary(o, t, mu1, mu2, mu3, pi2, pi3, rho2, rho3, ka2, om2, om3):

    """
        Mathys et al., 2011

    Implementation in DMpy works very slightly differently the the HGF toolbox. DMpy assumes that estimated value is
    estimated after seeing the outcome on the current trial, while the HGF toolbox reports predicted value for the
    current trial. To get the HGF to return the appropriate estimated value here, we add the estimated mu1 at the end
    of the update loop, which provides the estimated value after seeing the outcome on each trial.

    Args:
        o: Outcome
        mu1: Estimation of stimulus category (not used, DMpy just expects this input)
        mu2: Estimation of stimulus tendency toward 0 or 1
        mu3: Estimation of volatility
        pi2: Precision of second level estimation
        pi3: Precision of third level estimation
        rho2:
        rho3:
        ka2:
        om2:
        om3:

    Returns:

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

    # add mu (this is the updated muhat1 after seeing the outcome on this trial, and is necessary for DMpy)
    mu1 = T.nnet.sigmoid(mu2 + t * rho2)

    return (mu1, mu2, mu3, pi2, pi3, da1, da2, da3)


def dual_lr_qlearning(o, t, v, alpha_p, alpha_n):

    """
    Dual learning rate Q-learning model, from Palminteri et al., 2017, Nature Human Behaviour

    Args:
        o: Trial outcome
        v: Value on previous trial
        alpha_p: Learning rate for positive prediction errors
        alpha_n: Learning rate for negative prediction errors

    Returns:
        value: Value on current trial
        pe: prediction error
        weighted_pe: prediction error weighted by learning rate
    """

    pe = o - v

    weighted_pe = T.switch(T.lt(pe, 0), alpha_n * pe, alpha_p * pe)

    value = v + weighted_pe

    return (value, pe, weighted_pe)


def uncertainty_dlr(o, t, v, alpha, beta):

    """
    Dynamic learning rate model where learning rate varies depending on magnitude of squared prediction errors

    Args:
        o: Outcome
        t: Trial
        v: Value from previous trial
        alpha: Previous learning rate
        beta: Free parameter governing the degree to which the learning date changes on each trial

    Returns:

    """

    pe = o - v
    value = v + alpha * pe

    alpha_m = alpha + beta * (T.pow(pe, 2) - alpha)

    return (value, alpha_m, pe)


