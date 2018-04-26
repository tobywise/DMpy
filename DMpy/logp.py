import numpy as np
import theano.tensor as T
import theano
import pymc3 as pm
from DMpy.utils import beta_response_transform, beta_response_transform_t


def r2(true, predicted):

    if not T.eq(true.shape, predicted.shape):
        try:
            raise AttributeError("True and predicted arrays should have the same shape, current shapes: True = {0},"
                                 " predicted = {1}".format(true.shape, predicted.shape))
        except:
            raise AttributeError("True and predicted arrays should have the same shape")

    else:
        sst = T.power(true - true.mean(), 2).sum()

        ssr = T.power(true - predicted, 2).sum()

        r2 = T.switch(T.eq(sst, 0), 1, 1 - ssr / sst)

    return r2


def r2_multiplied(true, predicted):

    return r2(true, predicted) * 100000


def mse(true, predicted):

    if not T.eq(true.shape, predicted.shape):
        try:
            raise AttributeError("True and predicted arrays should have the same shape, current shapes: True = {0},"
                                 " predicted = {1}".format(true.shape, predicted.shape))
        except:
            raise AttributeError("True and predicted arrays should have the same shape")

    else:
        mse_score = T.power(true - predicted, 2).mean()

    return mse_score


def rss(true, predicted):

    rss = T.power(true - predicted, 2).sum()

    return -rss


def log_likelihood(true, predicted):

    return (np.log(predicted[true.nonzero()]).sum() +
     np.log(1 - predicted[(1 - true).nonzero()]).sum())


def normal_likelihood(mu=None, sd_mu=0, sd_sd=100 ** 2):

    sd = T.exp(pm.Normal('sd', mu=sd_mu, sd=sd_sd))

    return pm.Normal.dist(mu=mu, sd=sd)


def beta_likelihood(mu=None, phi_mu=0, phi_sd=100 ** 2):

    phi = T.exp(pm.Normal('phi', mu=phi_mu, sd=phi_sd, testval=1))

    return pm.Beta.dist(alpha=mu * phi, beta=(1 - mu) * phi)

# TODO bernoulli
def bernoulli_likelihood(p):

    p = beta_response_transform_t(p)

    return pm.Bernoulli.dist(p=p)