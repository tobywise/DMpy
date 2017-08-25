import numpy as np
import pymc3 as pm
from pymc3.distributions import Continuous
from theano import scan, printing, function
import theano.tensor as T
from pymc3 import fit, sample_approx
from pymc3 import traceplot, dic, summary, find_MAP
import matplotlib.pyplot as plt
import warnings
import re
import pandas as pd
import seaborn as sns
from timeit import default_timer as timer
from collections import OrderedDict
from itertools import product, combinations
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
from DMpy.utils import generate_pymc_distribution, n_returns, function_wrapper, load_data, load_outcomes, \
    parameter_table, model_fit, generate_choices2, n_obs_dynamic, simulated_responses

sns.set_style("white")
sns.set_palette("Set1")

# TODO parameter recovery doesn't work with hierarchical estimates as there is a different parameter for each subject

# TODO different priors/simulation parameter values for each subject
# TODO bayes optimal
# TODO move hierarchical setting to parameters - allow more detail specification and combinations of h/non-h
# TODO different outcome lists for different subjects
# TODO more than 2 stimuli - outcome/response for each
# TODO missed responses
# TODO reordering function arguments for scan
# TODO cauchy distribution
# TODO subject specific model fits (DIC etc)
# TODO combinations of priors (Oli) - lists of parameters, automatic combination
# TODO parallelisation


def _initialise_parameters(learning_parameters, observation_parameters, n_subjects, mle, hierarchical):

    """
    Assigns PyMC3 distributions to provided parameters. PyMC3 distribution is stored in the .pymc_distribution attribute

    Args:
        learning_parameters: Parameters for the learning model
        observation_parameters: Parameters for the observation model
        n_subjects: Number of subjects, used for determining shape of parameter arrays during fitting
        mle: MLE flag - if true, parameters are assigned uniform/flat priors
        hierarchical: Hierarchical flag - if true, hierarchical priors are added

    Returns:
        dynamic_parameters: list of dynamic parameters with distributions assigned
        static_parameters: list of static parameters with distributions assigned
        observation_parameters: the provided observation parameters, there's probably a reason for this but I can't remember
                                it
    """

    if type(learning_parameters) is not list:
        learning_parameters = [learning_parameters]

    learning_parameter_names = []

    for n, p in enumerate(learning_parameters):
        learning_parameters[n] = generate_pymc_distribution(p, n_subjects=n_subjects, mle=mle,
                                                                 hierarchical=hierarchical)
        learning_parameter_names.append(p.name)

    dynamic_parameters = []
    static_parameters = []

    for p in learning_parameters:  # add parameters to static or dynamic lists

        if not hasattr(p, 'pymc_distribution'):
            raise AttributeError("Parameter {0} has no distribution information, make sure at least" \
                                 " the mean is specified".format(p.name))

        if p.dynamic:
            dynamic_parameters.append(p.pymc_distribution)
        else:
            static_parameters.append(p.pymc_distribution)

    for n, p in enumerate(observation_parameters):

        if not p.name in learning_parameter_names:
            observation_parameters[n] = generate_pymc_distribution(p, n_subjects=n_subjects, mle=mle,
                                                                        hierarchical=hierarchical)
        else:  # awkward - would be better to make shared list of parameters first then split into l/obs lists
            for i in learning_parameters:
                if i.name == p.name:
                    observation_parameters[n] = i

    return dynamic_parameters, static_parameters, observation_parameters


def _recovery(parameter_table, sims):
    
    sns.set_palette("deep")
    sims = sims.reset_index(drop=True)
    fit_params = [i for i in parameter_table.columns if not 'sd_' in i and not 'Subject' in i]
    parameter_table = pd.merge(parameter_table, sims, on='Subject')
    print "Performing parameter recovery tests..."
    p_values = []
    p_values_sim = []
    n_p_free = len(fit_params)
    print n_p_free
    f, axarr = plt.subplots(1, n_p_free, figsize=(3 * n_p_free, 3.5))
    for n, p in enumerate(fit_params):  # this code could all be made far more efficient
        print n
        if p.replace('mean_', '') + '_sim' not in sims.columns:
            raise ValueError("Simulated values for parameter {0} not found in response file".format(p))
        p_values.append(parameter_table[p])
        p_values_sim.append(parameter_table[p.replace('mean_', '') + '_sim'])
        if n_p_free > 1:
            ax = axarr[n]
        else:
            ax = axarr
        sns.regplot(parameter_table[p.replace('mean_', '') + '_sim'], parameter_table[p], ax=ax)
        ax.set_xlabel('Simulated {0}'.format(p), fontweight='bold')
        ax.set_ylabel('Estimated {0}'.format(p), fontweight='bold')
        ax.set_title('Parameter {0}'.format(p), fontweight='bold')
        
        sim_min = np.min(parameter_table[p.replace('mean_', '') + '_sim'])
        sim_max = np.max(parameter_table[p.replace('mean_', '') + '_sim'])
        true_min = np.min(parameter_table[p])
        true_max = np.max(parameter_table[p])
        ax.set_xlim([sim_min - np.abs(sim_min)/10., sim_max + np.abs(sim_max)/10.])
        ax.set_ylim([true_min - np.abs(true_min)/10., true_max + np.abs(true_max)/10.])
        
        sns.despine()
    plt.tight_layout()
    cor = np.corrcoef(p_values, p_values_sim)[n_p_free:, :n_p_free]
    fig, ax = plt.subplots(figsize=(n_p_free * 2, n_p_free * 1.8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(cor, cmap=cmap, square=True, linewidths=.5, xticklabels=fit_params,
                yticklabels=fit_params, annot=True)  # order might not work here
    ax.set_xlabel('Simulated', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')

    return cor
    


class PyMCModel(Continuous):

    """
    Instance of PyMC3 model used to fit models

    """

    def __init__(self, learning_models, learning_parameters, observation_model, observation_parameters, responses, hierarchical,
                 outcomes, mle=False, *args, **kwargs):
        super(PyMCModel, self).__init__(*args, **kwargs)

        self.fit_complete = False

        # Define parameter distributions

        self.learning_model = learning_models[0]
        self.__learning_model_initial = learning_models[1]
        self.observation_model = observation_model
        self.learning_parameters = learning_parameters
        self.observation_parameters = observation_parameters[0]
        self.__observation_dynamic_inputs = observation_parameters[1]
        self.outcomes = outcomes
        
        # check responses etc

        if len(responses.shape) > 1 or (len(responses.shape) == 2 and responses.shape[0] > 1):
            self.n_subjects = responses.shape[0]
        else:
            self.n_subjects = 1

        self.responses = responses.astype(int)
        if self.responses.shape[0] == 1:
            self.responses = self.responses[0]

        if responses.shape[1] != len(outcomes):
            raise ValueError("Responses ({0}) and outcomes ({1}) have unequal lengths".format(responses.shape[1],
                                                                                              len(outcomes)))
        
        ## initialise parameters
        
        self.dynamic_parameters, self.static_parameters, self.observation_parameters = \
            _initialise_parameters(self.learning_parameters, self.observation_parameters, self.n_subjects, mle,
                                  hierarchical)

        ## learning models with multiple outputs
        ## check number of dynamic parameters, if number of learning function outputs is longer, add nones to outputs info

        self.__n_dynamic = len(self.dynamic_parameters)
        self.__n_learning_returns, _ = n_returns(self.learning_model)


    def get_value(self, x):

        """
        Function to run the learning and observation models on the provided outcome data

        Need to make sure order of arguments is right
        OR figure out a way to match scan arguments to function arguments


        Args:
            x: Outcome data

        Returns:
            prob: Probability of choosing an option
        """

        # begin awful hack - there must be a better way to get values on trial+1 while retaining initial value on t = 0

        try:
            value, _ = scan(fn=self.learning_model,
                            sequences=dict(input=x, taps=[-1]),
                            outputs_info=self.dynamic_parameters + [None] * (self.__n_learning_returns - self.__n_dynamic),
                            non_sequences=self.static_parameters)

            _value, _ = scan(fn=self.__learning_model_initial,  # THIS BIT COULD EASILY BE DONE WITHOUT SCAN
                                    sequences=dict(input=x[0:2]),  # just need to make sure the output is correct for joining
                             outputs_info=self.dynamic_parameters + [None] * (self.__n_learning_returns - self.__n_dynamic),
                             non_sequences=self.static_parameters)
        except ValueError as e:
            if "None as outputs_info" in e.message:
                raise ValueError("Mismatch between number of dynamic outputs and number of dynamic inputs. \n"
                                 "Make sure function outputs and inputs match (i.e. all dynamic inputs have a corresponding\n"
                                 " returned value, and make sure dynamic parameters are correctly set to be dynamic and\n"
                                 " static parameters are set to be static")
            else:
                raise e

        if not len(value):
            value = [value]
            _value = [_value]

        value = value[:self.__n_dynamic]  # hack, for some reason non-reused outputs don't join properly

        for n, v in enumerate(value):
            value[n] = T.concatenate([_value[n], v])[1:]

        # end awful hack
        observation_dynamics = [value[i] for i in self.__observation_dynamic_inputs]

        prob = self.observation_model(*observation_dynamics +
                                       [i.pymc_distribution for i in self.observation_parameters])[0]

        return prob


    def logp(self, x):
        """

        Calls the get_value function and then calculates logp for the model based on estimated probabilities

        Args:
            x: Outcome data

        Returns:
            Logp

        """
        prob = self.get_value(x)
        ll = (np.log(prob[self.responses.T.nonzero()]).sum() + np.log(1 - prob[(1 - self.responses.T).nonzero()]).sum())
        return ll


class RLModel():

    def __init__(self, learning_model, learning_parameters, observation_model, observation_parameters, name=''):
        self.name = name
        self.learning_model = learning_model
        self.learning_parameters = learning_parameters
        self.observation_model = observation_model
        self.observation_parameters = observation_parameters
        self.__observation_dynamic_inputs = []
        self.trace = None
        self.simulated = []
        self._model = None
        self.fit_complete = False

        self.__n_learning_returns, self.__learning_returns = n_returns(self.learning_model)
        self.__n_observation_returns, self.__observation_returns = n_returns(self.observation_model)

        n_dynamic = 0

        for p in self.learning_parameters:
            try:
                if p.dynamic:
                    n_dynamic += 1
            except AttributeError:
                raise ValueError("One or more parameters are not instances of the RLpackage parameter class.\n"
                                 "Failed with parameter value {0}".format(p))

        self.__learning_model_initial = function_wrapper(self.learning_model, self.__n_learning_returns, n_dynamic)

        if type(self.observation_parameters) is not list:
            self.observation_parameters = [self.observation_parameters]

        self.__observation_dynamic_inputs = [i for i in self.observation_parameters if isinstance(i, str)]
        self.observation_parameters = [i for i in self.observation_parameters if not isinstance(i, str)]

        for n, i in enumerate(self.__observation_dynamic_inputs):
            for nn, j in enumerate(self.learning_parameters):
                if j.name == i:
                    self.__observation_dynamic_inputs[n] = nn  # get the index of the relevant learning parameter
        self.__observation_dynamic_inputs = [0] + self.__observation_dynamic_inputs  # add zero for value output

        if any([isinstance(i, str) for i in self.__observation_dynamic_inputs]):  # check this worked
            raise ValueError("Observation model dynamic inputs don't match with learning model parameter names")

        n_obs_params = len(self.observation_parameters)
        self.__n_obs_dynamic = n_obs_dynamic(self.observation_model, n_obs_params)


    def fit(self, outcomes, responses, fit_method='MLE', hierarchical=False, plot=True, fit_stats=False, recovery=True,
            fit_kwargs=None, sample_kwargs=None):

        """
        General fitting method, calls appropriate underlying fitting methods as necessary

        Args:
            outcomes: task outcomes (e.g. A rewarded or not)
            responses: subject responses, as a .txt or .csv file with columns ['Responses', 'Subject']
            fit_method: method to use for fitting, one of 'MLE', 'MAP', 'Variational', 'MCMC', 'mle', 'map',
                        'variational', 'mcmc'. Default = 'MLE'
            hierarchical: Whether to perform hierarchical fitting - has no effect for MLE or MAP estimation. Default =
                          False
            plot: For sampling methods, provide traceplots. Default = True
            fit_stats: Provide fit statistics. Provided because for some reason this can take a while for variational
                        and MCMC methods so it can be convenient to turn it off when testing. Default = False
            recovery: If simulated parameter values are provided in the response file, will calculate correlations between
                      simulated and estimated parameters and produce correlation plots to assess parameter recovery success.
                      Default = True
            fit_kwargs: Dictionary of keyword arguments passed to underlying MLE, MAP and variational fitting functions.
            sample_kwargs: Dictionary of keyword arguments passed to underlying variational and MCMC sampling functions.

        """

        allowed_methods = ['MLE', 'MAP', 'Variational', 'MCMC', 'mle', 'map', 'variational', 'mcmc']

        if fit_kwargs is None:
            fit_kwargs = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        if fit_method in ['MLE', 'mle']:
            self._fit_MLE(outcomes=outcomes, responses=responses, plot=plot, recovery=recovery, **fit_kwargs)
            
        elif fit_method in ['MAP', 'map']:
            self._fit_MAP(outcomes=outcomes, responses=responses, plot=True, recovery=recovery, **fit_kwargs)

        elif fit_method in ['variational', 'Variational']:
            self._fit_variational(outcomes=outcomes, responses=responses, plot=plot, hierarchical=hierarchical, recovery=recovery,
                                  fit_stats=fit_stats, fit_kwargs=fit_kwargs, sample_kwargs=sample_kwargs)

        elif fit_method in ['MCMC', 'mcmc']:
            self._fit_MCMC(outcomes=outcomes, responses=responses, plot=plot, hierarchical=hierarchical, recovery=recovery,
                           fit_stats=fit_stats, **sample_kwargs)

        else:
            raise ValueError("Invalid fitting method provided ({0}). Fit method should be one of {1}"
                             .format(fit_method, allowed_methods))


    def _fit_MCMC(self, outcomes, responses, hierarchical=False, plot=True, fit_stats=False, recovery=True, **kwargs):

        sns.set_palette("deep")

        print "Loading data"

        self.subjects, responses, sims = load_data(responses)
        n_subjects = len(self.subjects)

        outcomes = load_outcomes(outcomes)

        print "Fitting model using NUTS"
        start = timer()

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
            print "Performing non-hierarchical model fitting for {0} subjects".format(n_subjects)


        with pm.Model() as rl:
            m = PyMCModel('rw', learning_models=(self.learning_model, self.__learning_model_initial),
                          learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model,
                          observation_parameters=[self.observation_parameters, self.__observation_dynamic_inputs],
                          responses=responses, observed=outcomes, outcomes=outcomes, hierarchical=hierarchical)

            self.trace = pm.sample(**kwargs)

            if plot:
                traceplot(self.trace)

        self._model = rl
        self.fit_values = pm.df_summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()

        print "\nPARAMETER ESTIMATES\n"

        self.parameter_table = parameter_table(pm.df_summary(self.trace), self.subjects)
        print self.parameter_table

        if recovery and sims is not None:
            self.recovery_correlations = _recovery(self.parameter_table, sims)

        # these seem to take a lot of time...
        if fit_stats:
            print "Calculating DIC..."
            self.DIC = pm.dic(self.trace, rl)
            print "Calculating WAIC..."
            self.WAIC = pm.waic(self.trace, rl)[0]
            print "Calculated fit statistics"


        #self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)
        self.fit_complete = True
        end = timer()
        print "Finished model fitting in {0} seconds".format(end - start)


    def _fit_variational(self, outcomes, responses, plot=True, hierarchical=True, fit_stats=False, recovery=True,
                         fit_kwargs=None, sample_kwargs=None):

        sns.set_palette("deep")

        print "Loading data"

        self.subjects, responses, sims = load_data(responses)
        # responses = responses.T
        n_subjects = len(self.subjects)

        outcomes = load_outcomes(outcomes)

        print "\n-------------------" \
              "Fitting model using ADVI" \
              "-------------------\n"
        start = timer()
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
            m = PyMCModel('rw', learning_models=(self.learning_model, self.__learning_model_initial),
                          learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model,
                          observation_parameters=[self.observation_parameters, self.__observation_dynamic_inputs],
                          responses=responses, observed=outcomes, outcomes=outcomes, hierarchical=hierarchical)

            approx = fit(model=rl, **fit_kwargs)
            self.trace = sample_approx(approx, **sample_kwargs)

            print "Done"

            if plot:
                traceplot(self.trace)

        self.fit_values = pm.df_summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()
        self.fit_complete = True
        self._model = rl

        # elif method == 'MAP':  # need to transform these values
        #     for p, v in map_estimate.iteritems():
        #         if 'interval' in p:
        #             map_estimate[p] = backward(0, 1, v)
        #         elif 'lowerbound' in p:
        #             map_estimate[p] = np.exp(v)
        #     self.fit_values = self.trace

        self.logp = rl.logp
        self.vars = rl.vars

        print "\nPARAMETER ESTIMATES\n"

        self.parameter_table = parameter_table(pm.df_summary(self.trace), self.subjects)
        print self.parameter_table

        print recovery

        if recovery and sims is not None:
            self.recovery_correlations = _recovery(self.parameter_table, sims)

        if fit_stats:
            # these seem to take a lot of time...
            print "Calculating DIC..."
            self.DIC = pm.dic(self.trace, rl)
            print "Calculating WAIC..."
            self.WAIC = pm.waic(self.trace, rl)[0]
            print "Calculated fit statistics"

        # TODO figure out fit statistics for multi-subject model fits
        # self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)
        end = timer()
        print "Finished model fitting in {0} seconds".format(end - start)


    def _fit_MAP(self, outcomes, responses, plot=True, mle=False, recovery=True, **kwargs):

        print "Loading data"

        self.subjects, responses, sims = load_data(responses)
        n_subjects = len(self.subjects)

        outcomes = load_outcomes(outcomes)

        if mle:
            print "\n-------------------" \
                  "Finding MLE estimate" \
                  "-------------------\n"
        else:
            print "\n-------------------" \
                  "Finding MAP estimate" \
                  "-------------------\n"

        start = timer()

        # check data is correct TODO change for multi-subject data
        # assert len(outcomes) == len(observed), "Outcome and observed data are " \
        #                                        "different lengths ({0} and ({1}".format(len(outcomes), len(observed))

        print "Performing model fitting for {0} subjects".format(n_subjects)

        self._model = {}
        self.map_estimate = {}

        with pm.Model() as rl:

            m = PyMCModel('rw', learning_models=(self.learning_model, self.__learning_model_initial),
                          learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model,
                          observation_parameters=[self.observation_parameters, self.__observation_dynamic_inputs],
                          responses=responses, observed=outcomes, outcomes=outcomes, hierarchical=False, mle=mle)

            params = (m.distribution.learning_parameters, m.distribution.observation_parameters)
            params = [i for j in params for i in j]

            try:
                self.map_estimate = find_MAP()
            except ValueError as err:
                warnings.warn("Fitting failed, this is probably because your model returned NaN values")
                raise err

            self.raw_fit_values = self.map_estimate

            self._model = rl

        # print self.raw_fit_values

        self.fit_complete = True
        # need to backwards transform these values

        untransformed_params = {}

        for p in params:
            for m in self.raw_fit_values.keys():
                n = re.search('.+(?=_.+__)', m)
                if (n and n.group() == p.name) or m == p.name:
                    if '__' in m:
                        untransformed_params[p.name] = p.backward(self.raw_fit_values[m]).eval()
                    else:
                        untransformed_params[p.name] = self.raw_fit_values[m]

        self.fit_values = untransformed_params

        print "\nPARAMETER ESTIMATES\n"

        if n_subjects > 1:
            self.parameter_table = pd.DataFrame(self.fit_values)
            self.parameter_table['Subject'] = self.subjects
            self.parameter_table.sort_values('Subject')
            print self.parameter_table
        else:
            self.parameter_table = pd.DataFrame(data=[self.subjects + self.fit_values.values()])
            self.parameter_table.columns = ['Subject'] + self.fit_values.keys()

        if recovery and sims is not None:
            self.recovery_correlations = _recovery(self.parameter_table, sims)

        print self.parameter_table

        # self.logp = rl.logp
        # self.vars = rl.vars
        self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.map_estimate, rl.vars, outcomes)  # TODO MULTI SUBJECT

        self.DIC = None
        self.WAIC = None
        end = timer()
        print "Finished model fitting in {0} seconds".format(end-start)


    def _fit_MLE(self, outcomes, responses, plot=True, recovery=True, **kwargs):

        self._fit_MAP(outcomes=outcomes, responses=responses, plot=plot, mle=True, recovery=recovery, **kwargs)


    def tracePlot(self):
        sns.set_palette("deep")
        traceplot(self.trace)


    def simulate(self, outcomes, learning_parameters=None, observation_parameters=None, plot=False, responses=None,
                 plot_choices=False, return_prob=False, plot_outcomes=True, response_file='', n_subjects=1, correlations=False,
                 plot_variance=False, plot_correlations=False, plot_value=True):

        self.sim_learning_parameters = OrderedDict(learning_parameters)
        self.sim_observation_parameters = OrderedDict(observation_parameters)

        sim_dynamic = []
        sim_static = []
        sim_observation = []
        sim_observation_dynamic = []

        outcomes = load_outcomes(outcomes)

        if self.sim_learning_parameters == None and self.sim_observation_parameters == None:
            raise ValueError("Must explicitly provide parameter values for simulation")

        # create parameter combinations
        list_params = []  # list of parameters with multiple values, useful for stuff later on

        for p, v in self.sim_learning_parameters.iteritems():  # convert any single values to list
            if hasattr(v, '__len__'):
                list_params.append(p)
            else:
                self.sim_learning_parameters[p] = [v]

        p_combinations = np.array(list(product(*self.sim_learning_parameters.values()))) # get product
        p_combinations = p_combinations.repeat(n_subjects, axis=0)  # repeat x n_subjects

        for n, (p, v) in enumerate(self.sim_learning_parameters.iteritems()):
            self.sim_learning_parameters[p] = p_combinations[:, n]
        # each parameter now has a list of values

        # set up parameters
        for n, i in enumerate(self.learning_parameters):
            match = False
            for p, v in self.sim_learning_parameters.iteritems():
                if p == i.name:
                    if i.dynamic:
                        sim_dynamic.append(np.float64(v))
                    else:
                        sim_static.append(np.float64(v))
                    match = True
            if not match:
                raise ValueError("Parameter {0} has no value provided".format(i.name))

        for i in self.observation_parameters:
            match = False
            for p, v in self.sim_observation_parameters.iteritems():
                if i.name == p:
                    sim_observation.append(np.ones(p_combinations.shape[0]) * v)
                    match = True
            if not match:
                raise ValueError("Parameter {0} has no value provided".format(i.name))

        # generate row names for output
        rnames = []
        rnames_short = []

        for i in range(0, p_combinations.shape[0]):
            rnames_short.append(str(p_combinations[i, :]))
            rname = [str(x) for t in zip(self.sim_learning_parameters.keys(), p_combinations[i, :]) for x in t]
            rnames.append('.'.join(rname) + '_{0}'.format(i))

        # simulate with scan
        #
        outcomes = np.array(outcomes)
        temp_outcomes = np.hstack([outcomes, 2])  # add an extra outcome to get outputs for the final trial if needed (e.g. PE)

        value, updates = scan(fn=self.learning_model,
                              sequences=dict(input=temp_outcomes, taps=[-1]),
                              # sim_dynamic = fed back into next step, none for other things like PEs
                              outputs_info=sim_dynamic + [None] * (self.__n_learning_returns - len(sim_dynamic)),
                              non_sequences=sim_static)

        _value, _updates = scan(fn=self.__learning_model_initial,
                                sequences=dict(input=outcomes[0:2]),
                                outputs_info=sim_dynamic + [None] * (self.__n_learning_returns - len(sim_dynamic)),
                                non_sequences=sim_static)

        if not len(value):
            value = [value]
            _value = [_value]

        for n, v in enumerate(value):
            if n < len(sim_dynamic):
                value[n] = T.concatenate([_value[n], v[:-1]])[1:]

        observation_dynamics = [value[i] for i in self.__observation_dynamic_inputs]

        prob, obs_outs = self.observation_model(*observation_dynamics + sim_observation)
        prob = prob.eval()  # using .eval() is bad

        eval_value = [v.eval() for v in value] + [o.eval() for o in obs_outs]

        return_names = self.__learning_returns + self.__observation_returns[1:]
        result_dict = dict(zip(return_names, eval_value))
        result_dict['P'] = prob

        self.simulated.append((self.sim_learning_parameters, self.sim_observation_parameters, result_dict))

        return_names = [i.replace('[', '') for i in return_names]
        return_names = [i.replace(']', '') for i in return_names]

        if plot:

            if len(prob.shape) < 2:
                sns.set_palette(sns.color_palette(['#348ABD']))
            else:
                sns.set_palette("Blues")

            fig, axarr = plt.subplots(len(eval_value), 1, figsize=(8, 1.5 * len(eval_value)))

            if plot_value:
                axarr[0].plot(eval_value[0], color='#add0e4', label='V')
            if len(prob.shape) < 2:
                prob = prob.reshape((prob.shape[0], 1))
            for n in range(0, prob.shape[1]):
                axarr[0].plot(prob[:, n], label=str(p_combinations[n]))
            axarr[0].set_title('Choice probability', fontweight='bold')


            if plot_choices:
                axarr[0].scatter(np.arange(0, len(prob)), generate_choices2(prob), color='#72a23b', alpha=0.5,
                                 label='Simulated choices')
            if responses is not None:
                axarr[0].scatter(np.arange(0, len(prob)), responses, color='#f18f01', alpha=0.5, label='Observations')
            if plot_outcomes:
                axarr[0].scatter(np.arange(0, len(prob)), outcomes, color='#a23b72', alpha=0.5,
                                 label='Outcomes')

            axarr[0].set_xlim(0, len(prob))
            axarr[0].set_ylim(np.min(outcomes) - 0.5, np.max(outcomes) + 0.2)
            if prob.shape[1] < 8:
                axarr[0].legend()

            for i in range(1, len(eval_value)):
                axarr[i].plot(eval_value[i])
                axarr[i].set_xlim(0, len(prob))
                axarr[i].set_title(return_names[i], fontweight='bold')

            plt.tight_layout()

        if correlations:

            if len(prob.shape) < 2:
                warnings.warn("Must provide multiple parameter values for correlations, skipping")

            else:
                cor = np.corrcoef(prob.T)
                self.simulated_timeseries_correlation = np.mean(cor)

                print "Mean correlation between simulated timeseries = {0}".format(self.simulated_timeseries_correlation)

                if plot_correlations:

                    fig, axarr = plt.subplots(figsize=(11, 9))
                    cmap = sns.diverging_palette(220, 10, as_cmap=True)
                    sns.heatmap(cor, cmap=cmap, square=True, linewidths=.5, xticklabels=rnames_short, yticklabels=rnames_short,
                                annot=False)
                    plt.yticks(rotation=0)
                    plt.xticks(rotation=90)
                    plt.title("Parameters = {0}".format(self.sim_learning_parameters.keys()), fontweight='bold')

                    plt.tight_layout()

                    sns.set_palette("deep")

                    plt.figure()
                    sns.kdeplot(cor.flatten(), shade=True)
                    plt.tight_layout()

        if plot_variance:

            if not len(list_params):
                warnings.warn("No parameters have more than one value, skipping variance plots")

            else:

                # awkward method
                sns.set_palette("deep")

                prob_df = pd.DataFrame(prob.T)  # convert probs to dataframe (allows grouped means etc)
                prob_df = pd.concat([prob_df, pd.DataFrame(self.sim_learning_parameters)], axis=1)  # add parameters

                fig, axarr = plt.subplots(len(list_params) + 2, 1, figsize=(10, 2 * (len(list_params) + 2)))

                sensitivity_dict = OrderedDict()

                for n, p in enumerate(list_params):
                    temp_df = prob_df.groupby(p)
                    cols = [i for i in prob_df.columns if type(i) != str]
                    temp_df = temp_df[cols]
                    mean = temp_df.mean()  # gives mean per level of p
                    mean['run'] = np.arange(0, len(mean), 1)
                    mean = pd.melt(mean, value_vars=[c for c in mean.columns if c != 'run'], id_vars='run')

                    sns.tsplot(time='variable', value='value', unit='run', data=mean, ax=axarr[n],
                               err_style="unit_traces")
                    axarr[n].set_title(p, fontweight='bold')

                    if plot_outcomes:
                        axarr[n].scatter(np.arange(0, len(prob)), outcomes, facecolors='white', edgecolors='#696969',
                                         linewidth=1, label='Outcomes', s=5)

                    sensitivity_dict[p] = []  # for sensitivity analysis

                # attempt at a variance-based sensitivity analysis (no existing methods deal with timeseries models)
                # take variance in probability at each time point, use regression to see which parameters
                # explain most variance

                for i in range(0, len(outcomes)):
                    y = prob_df[i]
                    X = prob_df[list_params]
                    X = add_constant(X)
                    model = OLS(y, X)
                    results = model.fit()
                    for p in list_params:
                        sensitivity_dict[p].append(results.params[p])

                for p in list_params:
                    axarr[-2].plot(sensitivity_dict[p], marker=None, label=p)
                    axarr[-2].set_title('Variance explained', fontweight='bold')
                c_ylim = np.array(axarr[-2].get_ylim())
                c_ylim[1] += np.abs(c_ylim[0] - c_ylim[1]) / 3
                axarr[-2].set_ylim(c_ylim)
                axarr[-2].legend(ncol=len(list_params), loc='upper left', bbox_to_anchor=(0, 1))
                axarr[-2].set_xlim((0, len(outcomes)))

                # if plot_outcomes:
                #     axarr[-2].scatter(np.arange(0, len(prob)), outcomes, facecolors='white', edgecolors='#696969',
                #                      linewidth=1, label='Outcomes', s=5)

                print "Correlation between beta timeseries"
                print pd.DataFrame(sensitivity_dict).corr()

                sensitivity_array = np.array(sensitivity_dict.values())
                sensitivity_array = np.diff(sensitivity_array, axis=1)
                print sensitivity_array.shape

                for i in range(sensitivity_array.shape[0]):
                    axarr[-1].plot((sensitivity_array[i, :] - sensitivity_array[i, :].min()) / sensitivity_array[i, :].max(),
                                   label=sensitivity_dict.keys()[i])
                c_ylim = np.array(axarr[-1].get_ylim())
                c_ylim[1] += np.abs(c_ylim[0] - c_ylim[1]) / 3
                axarr[-1].set_ylim(c_ylim)
                axarr[-1].legend(ncol=len(list_params), loc='upper left', bbox_to_anchor=(0, 1))
                axarr[-1].set_title("Beta timeseries direction", fontweight='bold')
                axarr[-1].set_xlim((0, len(outcomes)))

                # if plot_outcomes:
                #     axarr[-1].scatter(np.arange(0, len(prob)), outcomes, facecolors='white', edgecolors='#696969',
                #                      linewidth=1, label='Outcomes', s=5)

                plt.tight_layout()

        choices = generate_choices2(prob)

        print "Finished simulating"
        sns.set_palette("deep")

        if len(response_file):
            if return_prob:
                simulated_responses(prob, rnames, response_file,
                                    (self.sim_learning_parameters.keys(), p_combinations),
                                    (self.sim_observation_parameters.keys(), np.array(sim_observation)))
            else:
                simulated_responses(choices, rnames, response_file,
                                    (self.sim_learning_parameters.keys(), p_combinations),
                                    (self.sim_observation_parameters.keys(), np.array(sim_observation)))
            return response_file
        elif return_prob:
            return prob
        else:
            return choices[0]


class Parameter():

    def __init__(self, name, distribution, lower_bound=None, upper_bound=None, mean=1., variance=None, dynamic=False,
                 **kwargs):

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
        self.__pymc_kwargs = kwargs

        if lower_bound:
            self.lower_bound = lower_bound
        if upper_bound:
            self.upper_bound = upper_bound

        self.mean = mean
        self.variance = variance

        self.transform_method = None


    def transform(self, name, transform):

        new_parameter = Parameter(self.name, self.distribution, self.lower_bound, self.upper_bound, self.mean,
                                  self.variance, self.dynamic)

        new_parameter.name = name

        new_parameter.transform_method = transform

        return new_parameter



