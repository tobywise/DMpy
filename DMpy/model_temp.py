import numpy as np
import pymc3 as pm
from pymc3.distributions import Continuous
from theano import scan
from pymc3 import Model, Normal, HalfNormal, DensityDist, Potential, Bound, Uniform, fit, sample_approx
from pymc3 import traceplot, dic, summary, find_MAP
import matplotlib.pyplot as plt
import warnings
import inspect
import re
import pandas as pd
import seaborn as sns
from theano.tensor import cast
sns.set_style("white")
sns.set_palette("Set1")

# TODO reordering function arguments for scan
# TODO cauchy/uniform distributions
# TODO MAP fitting - indiviudal parameters are the same
# TODO MLE fitting
# TODO subject specific model fits
# TODO combinations of priors (Oli) - lists of parameters, automatic combination


def get_backwward_transform(bounded_parameter):

    return bounded_parameter.pymc_distribution.transformation.backward


def generate_pymc_distribution(p, n_subjects=None, hierarchical=False):

    """
    Turns parameters into pymc3 parameter distributions for model fitting
    """

    if p.fixed:
        p.pymc_distribution = np.float64(np.ones(n_subjects) * p.mean)

    else:

        if hierarchical and n_subjects < 1:
            raise ValueError("Hierarchical parameters only possible with > 1 subject")

        if p.distribution == 'normal' and p.lower_bound is not None and p.upper_bound is not None:
            BoundedNormal = Bound(Normal, lower=p.lower_bound, upper=p.upper_bound)
            if hierarchical:
                p.pymc_distribution = BoundedNormal(p.name,
                                                    mu=BoundedNormal(p.name + '_group_mu', mu=p.mean, sd=p.variance),
                                                    sd=Uniform(p.name + '_group_sd', lower=0, upper=100),
                                                    shape=n_subjects)
            elif n_subjects > 1:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects)  # TODO doesn't work
            else:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance)
            p.backward = get_backwward_transform(p)

        elif p.distribution == 'normal' and p.lower_bound is not None:
            BoundedNormal = Bound(Normal, lower=p.lower_bound)
            if hierarchical:
                p.pymc_distribution = BoundedNormal(p.name,
                                                    mu=BoundedNormal(p.name  + '_group_mu', mu=p.mean, sd=p.variance),
                                                    sd=Uniform(p.name  + '_group_sd', lower=0, upper=100),
                                                    shape=n_subjects)
            elif n_subjects > 1:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects)
            else:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance)
            p.backward = get_backwward_transform(p)

        elif p.distribution == 'normal':
            if hierarchical:
                p.pymc_distribution = Normal(p.name,
                                             mu=Normal(p.name  + '_group_mu', mu=p.mean, sd=p.variance),
                                             sd=Uniform(p.name + '_group_sd', lower=0, upper=100),
                                             shape=n_subjects)
            elif n_subjects > 1:
                p.pymc_distribution = Normal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects)
            else:
                p.pymc_distribution = Normal(p.name, mu=p.mean, sd=p.variance)

    return p


def generate_choices2(pa):

    """
    Simulates choices based on choice probabilities
    """

    return (np.random.random(len(pa)) < pa).astype(int)


def backward(a, b, x):
    a, b = a, b
    r = (b - a) * np.exp(x) / (1 + np.exp(x)) + a
    return r


def model_fit(logp, fit_values, vars, outcome):

    """
    Calculates model fit statistics (log likelihood, BIC, AIC)
    """

    log_likelihood = logp(fit_values)
    BIC = len(vars) * np.log(len(outcome)) - 2. * log_likelihood  # might be the BIC
    AIC = 2. * (len(vars) - log_likelihood)

    return log_likelihood, BIC, AIC


def parameter_table(df_summary, subjects):

    """
    Attempts to turn the pymc3 output into a nice table of parameter values for each subject
    """

    df_summary = df_summary[['mean', 'sd']]
    df_summary = df_summary.reset_index()
    df_summary = df_summary[~df_summary['index'].str.contains('group')]

    n_subjects = len(subjects)
    n_parameters = len(df_summary) / n_subjects
    subject_column = pd.Series(subjects * n_parameters)
    df_summary['Subject'] = subject_column.values
    df_summary['index'] = pd.Series([re.search('.+(?=__)', i).group() for i in df_summary['index']]).values

    df_summary = df_summary.pivot(index='Subject', columns='index')
    df_summary.columns = df_summary.columns.map('_'.join)
    df_summary = df_summary.reset_index()

    return df_summary


def load_data(data_file):

    data = pd.read_csv(data_file)

    if 'Subject' not in data.columns or 'Response' not in data.columns:
        raise ValueError("Data file must contain the following columns: Subject, Response")

    n_subjects = len(np.unique(data['Subject']))

    if len(data) % n_subjects:
        raise AttributeError("Unequal number of trials across subjects")

    n_trials = len(data) / n_subjects

    if n_subjects > 1:
        print "Loading multi-subject data with {0} subjects".format(n_subjects)
    else:
        print "Loading single subject data"

    subjects = list(set(data.Subject))
    trial_index = np.tile(range(0, n_trials), n_subjects)
    data['Trial_index'] = trial_index

    data = data.pivot(columns='Subject', values='Response', index='Trial_index')
    data = data.values.T

    print "Loaded data, {0} subjects with {1} trials".format(n_subjects, n_trials)

    return subjects, data


def n_returns(f):

    return_code = inspect.getsourcelines(f)[0][-1].replace('\n', '')
    n = len(return_code.split(','))
    try:
        if n > 1:
            returns = re.search('(?<=\().+(?=\))', return_code).group().split(', ')
        else:
            returns = re.search('(?<=return ).+', return_code).group()
    except:
        warnings.warn("Could not retrieve function return names")
        returns = None

    return n, returns


class PyMCModel(Continuous):

    def __init__(self, learning_model, learning_parameters, observation_model, observation_parameters, responses, hierarchical,
                 outcomes, *args, **kwargs):
        super(PyMCModel, self).__init__(*args, **kwargs)

        self.fit = False

        # Define parameter distributions

        self.learning_model = learning_model
        self.observation_model = observation_model
        self.learning_parameters = learning_parameters
        self.outcomes = outcomes

        if type(self.learning_parameters) is not list:
            self.learning_parameters = [self.learning_parameters]

        if len(responses.shape) > 1 or (len(responses.shape) == 2 and responses.shape[0] > 1):
            self.n_subjects = responses.shape[0]
        else:
            self.n_subjects = 1

        for n, p in enumerate(self.learning_parameters):

            self.learning_parameters[n] = generate_pymc_distribution(p, n_subjects=self.n_subjects,
                                                                     hierarchical=hierarchical)

        self.dynamic_parameters = []
        self.static_parameters = []

        for p in self.learning_parameters:  # add parameters to static or dynamic lists

            if p.dynamic:  # parameter class needs this attribute
                self.dynamic_parameters.append(p.pymc_distribution)
            else:
                self.static_parameters.append(p.pymc_distribution)


        self.observation_parameters = observation_parameters
        if type(self.observation_parameters) is not list:
            self.observation_parameters = [self.observation_parameters]

        for n, p in enumerate(self.observation_parameters):

            self.observation_parameters[n] = generate_pymc_distribution(p, n_subjects=self.n_subjects,
                                                                        hierarchical=hierarchical)

        self.responses = responses.astype(int)

        ## learning models with multiple outputs
        ## check number of dynamic parameters, if number of learning function outputs is longer, add nones to outputs info

        self.__n_dynamic = len(self.dynamic_parameters)
        self.__n_learning_returns, _ = n_returns(self.learning_model)


    def get_value(self, x):

        # need to make sure order of arguments is right
        # OR figure out a way to match scan arguments to function arguments

        value, _ = scan(fn=self.learning_model,
                        sequences=[x],
                        outputs_info=self.dynamic_parameters + [None] * (self.__n_learning_returns - self.__n_dynamic),  # this might not work
                        non_sequences=self.static_parameters)

        if len(value):
            value = value[0]

        prob = self.observation_model(value, self.observation_parameters[0].pymc_distribution)  # TODO figure out how to use multiple parameters?

        return prob


    def logp(self, x):  # x = outcome data
        prob = self.get_value(x[0])
        print self.responses
        return np.log(prob[self.responses.nonzero()]).sum() + np.log(prob[(1 - self.responses).nonzero()]).sum()  # TODO might be an issue with multiple subjects here


class RLModel():

    def __init__(self, learning_model, learning_parameters, observation_model, observation_parameters):
        model_name = ''
        self.learning_model = learning_model
        self.learning_parameters = learning_parameters
        self.observation_model = observation_model
        self.observation_parameters = observation_parameters

        self.trace = None
        self.simulated = []

        self.fit = False


    def fit_NUTS(self, outcomes, responses, hierarchical=False, plot=True, **kwargs):

        print "Loading data"

        self.subjects, observed = load_data(observed)
        n_subjects = len(self.subjects)

        print "Fitting model using NUTS"

        # check data is correct
        # assert len(outcomes) == len(observed), "Outcome and observed data are " \
        #                                        "different lengths ({0} and ({1}".format(len(outcomes), len(observed))

        if hierarchical and n_subjects < 2:
            warnings.warn("\nWarning: Hierarchical model fitting only possible with more than one subject, "
                          "fitting individual subject\n")
            hierarchical = False

        elif hierarchical and n_subjects > 1:
            print "Performing hierarchical model fitting for {0} subjects".format(n_subjects)

        elif not hierarchical and n_subjects > 1:
            print "Performing non-hierarchical model fitting for {0} subjects - NOT IMPLEMENTED".format(n_subjects)


        with pm.Model() as rl:
            m = PyMCModel('rw', learning_model=self.learning_model, learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model, observation_parameters=self.observation_parameters,
                          responses=responses, observed=outcomes, hierarchical=hierarchical)

            self.trace = pm.sample(**kwargs)

            if plot:
                traceplot(self.trace)

        self.__model = rl
        self.fit_values = pm.df_summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()

        self.parameter_table = parameter_table(pm.df_summary(self.trace), self.subjects)
        print self.parameter_table

        self.DIC = pm.dic(self.trace, rl)
        self.WAIC = pm.waic(self.trace, rl)[0]

        #self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)
        self.fit = True


    def fit_variational(self, outcomes, responses, plot=True, draws=100, hierarchical=True, **kwargs):

        print "Loading data"

        self.subjects, observed = load_data(observed)
        n_subjects = len(self.subjects)

        print "\n-------------------" \
              "Fitting model using ADVI" \
              "-------------------\n"

        # check data is correct TODO change for multi-subject data
        # assert len(outcomes) == len(observed), "Outcome and observed data are " \
        #                                        "different lengths ({0} and ({1}".format(len(outcomes), len(observed))

        if hierarchical and n_subjects < 2:
            warnings.warn("\nWarning: Hierarchical model fitting only possible with more than one subject, "
                          "fitting individual subject\n")
            hierarchical = False

        elif hierarchical and n_subjects > 1:
            print "Performing hierarchical model fitting for {0} subjects".format(n_subjects)

        elif not hierarchical and n_subjects > 1:
            print "Performing non-hierarchical model fitting for {0} subjects".format(n_subjects)

        with pm.Model() as rl:
            m = PyMCModel('rw', learning_model=self.learning_model, learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model, observation_parameters=self.observation_parameters,
                          responses=responses, observed=outcomes, hierarchical=hierarchical)

            approx = fit(model=rl, **kwargs)
            self.trace = sample_approx(approx, draws=draws)

            if plot:
                traceplot(self.trace)

        self.fit_values = pm.df_summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()
        self.fit = True

        # elif method == 'MAP':  # need to transform these values
        #     for p, v in map_estimate.iteritems():
        #         if 'interval' in p:
        #             map_estimate[p] = backward(0, 1, v)
        #         elif 'lowerbound' in p:
        #             map_estimate[p] = np.exp(v)
        #     self.fit_values = self.trace

        self.logp = rl.logp
        self.vars = rl.vars

        self.parameter_table = parameter_table(pm.df_summary(self.trace), self.subjects)
        print self.parameter_table

        self.DIC = pm.dic(self.trace, rl)
        self.WAIC = pm.waic(self.trace, rl)[0]

        # TODO figure out fit statistics for multi-subject model fits
        # self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)


    def fit_MAP(self, outcomes, responses, plot=True, **kwargs):

        print "Loading data"

        self.subjects, responses = load_data(responses)
        n_subjects = len(self.subjects)

        print "\n-------------------" \
              "Finding MAP estimate" \
              "-------------------\n"

        # check data is correct TODO change for multi-subject data
        # assert len(outcomes) == len(observed), "Outcome and observed data are " \
        #                                        "different lengths ({0} and ({1}".format(len(outcomes), len(observed))

        print "Performing model fitting for {0} subjects".format(n_subjects)

        self._model = {}
        self.map_estimate = {}

        for n, s in enumerate(self.subjects):

            with pm.Model() as rl:

                print "Fitting {0}".format(s)

                m = PyMCModel('rw', learning_model=self.learning_model, learning_parameters=self.learning_parameters,
                              observation_model=self.observation_model, observation_parameters=self.observation_parameters,
                              responses=responses[n], observed=responses, outcomes=outcomes, hierarchical=False)

                params = (m.distribution.learning_parameters, m.distribution.observation_parameters)
                params = [i for j in params for i in j]

                self.map_estimate[s] = find_MAP()
                print self.map_estimate[s]

                if n == 0:
                    self.raw_fit_values = self.map_estimate[s]
                else:
                    for k, v in self.map_estimate[s].items():
                        self.raw_fit_values[k] = np.append(self.raw_fit_values[k], self.map_estimate[s][k])

                self._model[s] = rl

        print self.raw_fit_values

        self.fit = True
        # need to backwards transform these values
        untransformed_params = {}

        for p in params:
            for m in self.raw_fit_values.keys():
                if p.name in m:
                    untransformed_params[p.name] = p.backward(self.raw_fit_values[m]).eval()

        print untransformed_params
        self.fit_values = untransformed_params

        if n_subjects > 1:
            self.parameter_table = pd.DataFrame(self.fit_values)
            self.parameter_table['Subject'] = self.subjects
            print self.parameter_table
        else:
            self.parameter_table = pd.DataFrame(data=[self.subjects + self.fit_values.values()])
            self.parameter_table.columns = ['Subject'] + self.fit_values.keys()


        # self.logp = rl.logp
        # self.vars = rl.vars
        # self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.map_estimate, rl.vars, outcomes)  # TODO MULTI SUBJECT

        self.DIC = None
        self.WAIC = None


    def fit_MLE(self, outcomes, responses, plot=True, hierarchical=True, **kwargs):

        from scipy.optimize import minimize

        print "MLE"

        def model_func(data):

            m = PyMCModel('rw', learning_model=self.learning_model, learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model, observation_parameters=self.observation_parameters,
                          responses=responses, observed=responses, outcomes=outcomes, hierarchical=hierarchical)

            return(m.logp(data))

        minimize(model_func, responses, args=(outcomes))

        return "Not implemented"


    def tracePlot(self):

        traceplot(self.trace)


    def simulate(self, outcomes, learning_parameters=None, observation_parameters=None, plot=False, responses=None,
                 sim_choices=False):

        # TODO rework for hierarchical estimates - provide both group and individual parameter based simulations?

        sim_dynamic = []
        sim_static = []
        sim_observation = []

        ## TODO multi-subject observed data

        if not self.fit and learning_parameters == None and observation_parameters == None:
            raise ValueError("Model has not been fit, must explicitly provide parameter values for simulation")

        if not learning_parameters:

            learning_parameters = self.fit_values

        if not observation_parameters:

            observation_parameters = self.fit_values

        for p, v in learning_parameters.iteritems():

            for i in self.learning_parameters:
                if i.name == p:
                    if i.dynamic:
                        sim_dynamic.append(np.float64(v))
                    else:
                        sim_static.append(np.float64(v))

        for p, v in observation_parameters.iteritems():

            for i in self.observation_parameters:
                if i.name == p:
                    sim_observation.append(np.float64(v))

        ## learning models with multiple outputs
        ## check number of dynamic parameters, if number of learning function outputs is longer, add nones to outputs info

        self.__n_learning_returns, self.__learning_returns = n_returns(self.learning_model)

        # simulate with scan

        value, updates = scan(fn=self.learning_model,
                              sequences=[outcomes],
                              outputs_info=sim_dynamic + [None] * (self.__n_learning_returns - len(sim_dynamic)),
                              non_sequences=sim_static)

        if not len(value):
            value = [value]

        prob = self.observation_model(value[0], sim_observation[0])
        prob = prob.eval()
        eval_value = [v.eval() for v in value]

        result_dict = dict(zip(self.__learning_returns, eval_value))
        result_dict['P'] = prob

        self.simulated.append((learning_parameters, observation_parameters, result_dict))

        if plot:

            fig, axarr = plt.subplots(len(eval_value), 1)

            axarr[0].plot(prob, color='#348ABD', label='P')
            axarr[0].set_title('Choice probability', fontweight='bold')

            if sim_choices:
                axarr[0].scatter(np.arange(0, len(prob)), generate_choices2(prob), color='#a23b72', alpha=0.5,
                                 label='Simulated choices')
            if responses is not None:
                axarr[0].scatter(np.arange(0, len(prob)), responses, color='#f18f01', alpha=0.5, label='Observations')

            axarr[0].set_xlim(0, len(prob))
            axarr[0].set_ylim(np.min(outcomes) - 0.5, np.max(outcomes) + 0.2)
            axarr[0].legend(frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.03), loc='lower center')


            for i in range(1, len(value)):
                axarr[i].plot(eval_value[i], color='#348ABD')
                axarr[i].set_xlim(0, len(prob))
                axarr[i].set_title(self.__learning_returns[i], fontweight='bold')

            plt.tight_layout()

        if sim_choices:
            return generate_choices2(prob)
        else:
            return prob


class Parameter():

    def __init__(self, name, distribution, lower_bound=None, upper_bound=None, mean=1., variance=None, dynamic=False):

        if distribution == 'uniform' and (lower_bound == None or upper_bound == None):
            raise ValueError("Must specify upper and lower bounds for parameters with uniform distribution")

        elif distribution == 'fixed' and variance == None and lower_bound == None and upper_bound == None:
            self.fixed = True
        else:
            self.fixed = False

        self.name = name
        self.distribution = distribution
        self.dynamic = dynamic
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if lower_bound:
            self.lower_bound = lower_bound
        if upper_bound:
            self.upper_bound = upper_bound

        self.mean = mean
        self.variance = variance



