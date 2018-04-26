import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def model_comparison(models, measure='logp', individual=True):

    if not isinstance(models, list):
        raise TypeError("Models should be supplied as a list, {0} given".format(type(models)))

    if len(models) <2:
        raise ValueError("Must provide at least two models for model comparison, {0} models provided".format(len(models)))

    if not all([m.fit_complete for m in models]):
        raise AttributeError("At least one model has not been fit, ensure all models have been fit before comparing models")

    if np.any(np.diff([m.n_subjects for m in models]) > 0):
        raise AttributeError("Models have different numbers of subjects")

    if individual:

        individual_fit_stats = []

        for n, m in enumerate(models):
            m_fit = m.individual_fits()[['subject', measure]]
            if len(m.name):
                model_name = m.name
            else:
                model_name = 'Model {0}'.format(n)
            m_fit.columns = [m.replace(measure, model_name +' {0}'.format(measure)) for m in m_fit.columns]
            individual_fit_stats.append(m_fit)

        individual_fit_stats = pd.concat(individual_fit_stats, axis=1)

        return individual_fit_stats

