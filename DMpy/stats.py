import numpy as np
from pymc3.model import modelcontext


def dic(trace, model=None):
    """Calculate the deviance information criterion of the samples in trace from model
    Read more theory here - in a paper by some of the leading authorities on Model Selection -
    dx.doi.org/10.1111/1467-9868.00353

    Faster version of the PyMC3 DIC function - but requires individual subject likelihoods

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.

    Returns
    -------
    `float` representing the deviance information criterion of the model and trace
    """

    print "A"
    model = modelcontext(model)
    print "B"
    mean_deviance = -2 * np.mean([model.logp(pt) for pt in trace])
    print "C"
    free_rv_means = {rv.name: trace[rv.name].mean(
        axis=0) for rv in model.free_RVs}
    print "D"
    deviance_at_mean = -2 * model.logp(free_rv_means)
    print "E"
    return 2 * mean_deviance - deviance_at_mean
