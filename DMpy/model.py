import numpy as np
import pymc3 as pm
from pymc3.distributions import Continuous
from theano import scan, function, printing
import theano.tensor as T
import theano
from pymc3 import fit, sample_approx
from pymc3 import traceplot, find_MAP
import matplotlib.pyplot as plt
import warnings
import re
import pandas as pd
import seaborn as sns
from timeit import default_timer as timer
from collections import OrderedDict, Counter
from itertools import product
from DMpy.utils import *
from DMpy.logp import *
from sklearn.metrics import r2_score
import copy
import inspect

theano.config.compute_test_value = "off"

sns.set_style("white")
sns.set_palette("Set1")

# TODO infer logp method from data
# TODO use observation model only in cases where there is no learning!
# TODO BUG - changing logp method doesn't work without reloading model instance
# TODO create a data class to make things more modular and organised?
# TODO methods for model comparison, e.g. take a list of fits and produce plots (or individual subject plots)
# TODO methods for model recovery - could take multiple simulation *objects*?
# TODO more automated recovery - given parameter ranges could automatically generate range of parameter values
# TODO change simulated results to an object? I.e. create a simulation class and use this to store results and do things with them
# TODO other likelihood functions - make this modular like models
# TODO make model fit metrics modular

# TODO Check re-entered parameters are given as dynamic and non-reentered ones aren't - getting dynamic setting wrong is too easy and causes problems that aren't easy to spot/solve
# TODO change recovery to a method of the model class which is called manually after fitting

# TODO multiple outcome arrays (sequences are entered as a list so should be possible) - would allow for interesting things to be added to the model (e.g. other stimuli)

# TODO move hierarchical setting to parameters - allow more detail specification and combinations of h/non-h
# TODO more than 2 stimuli - outcome/response for each
# TODO cauchy distribution + others
# TODO subject specific model fits (DIC etc)


def _initialise_parameters(learning_parameters, observation_parameters, n_subjects, n_runs, mle, hierarchical):

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
        if not p.name in learning_parameter_names:
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

    if observation_parameters[0] is not None:
        for n, p in enumerate(observation_parameters):

            if not p.name in learning_parameter_names:
                observation_parameters[n] = generate_pymc_distribution(p, n_subjects=n_subjects, mle=mle,
                                                                            hierarchical=hierarchical)
            else:  # awkward - would be better to make shared list of parameters first then split into l/obs lists
                for i in learning_parameters:
                    if i.name == p.name:
                        observation_parameters[n] = i

    return dynamic_parameters, static_parameters, observation_parameters


def _add_noise(timeseries, mean, sd, lower_bound=0, upper_bound=1):

    noisy_timeseries = timeseries + np.random.normal(mean, sd, timeseries.shape)

    noisy_timeseries[noisy_timeseries > upper_bound] = upper_bound
    noisy_timeseries[noisy_timeseries < lower_bound] = lower_bound

    return noisy_timeseries


class _PyMCModel(Continuous):

    """
    Instance of PyMC3 model used to fit models. Used internally by DMModel class - do not use directly.
    """

    def __init__(self, learning_models, learning_parameters, observation_model, observation_parameters, responses, hierarchical,
                 n_subjects, time, n_runs, mle=False, outcomes=None, model_inputs=None, logp_function=None, vars=None,
                 logp_args=None, *args, **kwargs):
        super(_PyMCModel, self).__init__(*args, **kwargs)

        self.fit_complete = False
        self.learning_model = learning_models[0]
        self.__learning_model_initial = learning_models[1]
        self.observation_model = observation_model
        self.learning_parameters = learning_parameters
        self.observation_parameters = observation_parameters[0]
        self.__observation_dynamic_inputs = observation_parameters[1]
        self.n_subjects = n_subjects
        self.responses = responses
        self.outcomes = outcomes
        self.time = time
        self.n_runs = n_runs
        self.logp_function = logp_function
        self.model_inputs = model_inputs
        self.logp_args = logp_args
        self.vars = vars

        if self.logp_args is None:
            self.logp_args = []

        self.logp_args = copy.copy(self.logp_args)  # things get messy if the same logp args are used for different models

        for n, input in enumerate(self.model_inputs):
            self.model_inputs[n] = dict(input=input, taps=[-1])

        # TODO need to check for inconsistent number of model inputs and arguments in model function

        if len(np.unique(n_runs)) != 1:
            raise ValueError("All subjects must have the same number of runs")

        self.n_runs = self.n_runs.min()

        ## initialise parameters

        self.dynamic_parameters, self.static_parameters, self.observation_parameters = \
            _initialise_parameters(self.learning_parameters, self.observation_parameters, self.n_subjects, self.n_runs,
                                   mle, hierarchical)

        ## learning models with multiple outputs
        ## check number of dynamic parameters, if number of learning function outputs is longer, add nones to outputs info

        self.__n_dynamic = len(self.dynamic_parameters)
        self.__n_learning_returns, self.__learning_return_names = n_returns(self.learning_model)
        self.__n_learning_returns, self.__learning_return_names = n_returns(self.learning_model)
        if self.observation_model is not None:
            self.__n_observation_returns, self.__observation_return_names = n_returns(self.observation_model)

        self.responses = self.responses.T
        if self.responses.ndim == 1:
            self.responses = self.responses.reshape(self.responses.shape[0], 1)
        if self.logp_function == 'normal':
            # Assume that if there's no observation model we're using value as our response variable
            if self.observation_model is None:
                self.logp_args = {'mu': 'value'}
            else:
                self.logp_args = {'mu': 'prob'}
            self.logp_function = normal_likelihood
        elif self.logp_function == 'beta':
            if self.observation_model is None:
                self.logp_args = {'mu': 'value'}
            else:
                self.logp_args = {'mu': 'prob'}
            self.logp_function = beta_likelihood
        elif self.logp_function == 'bernoulli':
            self.logp_args = {'p': 'prob'}
            self.logp_function = bernoulli_likelihood


        if not isinstance(self.logp_args, dict):
            raise TypeError("Logp arguments should be supplied as a dictionary, supplied {0}".format(type(self.logp_args)))

        self.logp_distribution = None


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

        # begin awful hack - there is probably a better way to get values on trial+1 while retaining initial value on t = 0
        self.static_parameters_reshaped = [np.repeat(i, self.n_runs).astype('float64') for i in self.static_parameters]
        self.dynamic_parameters_reshaped = [np.repeat(i, self.n_runs).astype('float64') for i in self.dynamic_parameters]
        if self.observation_parameters[0] is not None:
            self.observation_parameters_reshaped = [np.repeat(i.pymc_distribution, self.n_runs) for i in self.observation_parameters]
        else:
            self.observation_parameters_reshaped = None

        time = T.ones_like(x) * T.arange(0, x.shape[0]).reshape((x.shape[0], 1))

        # print [dict(input=x, taps=[-1]), dict(input=time, taps=[-1])] + self.model_inputs

        model_inputs_initial = copy.deepcopy(self.model_inputs)
        for i in model_inputs_initial:
            i['input'] = i['input'][:1, :]
            i.pop('taps')

        try:
            value, _ = scan(fn=self.learning_model,
                            sequences=[dict(input=x, taps=[-1]), dict(input=time, taps=[-1])] + self.model_inputs,
                            outputs_info=self.dynamic_parameters_reshaped + [None] * (self.__n_learning_returns - self.__n_dynamic),
                            non_sequences=self.static_parameters_reshaped)
            _value, _ = scan(fn=self.__learning_model_initial,
                                    sequences=[dict(input=x[:1, :]), dict(input=time[:1, :])] + model_inputs_initial,
                             outputs_info=self.dynamic_parameters_reshaped + [None] * (self.__n_learning_returns - self.__n_dynamic),
                             non_sequences=self.static_parameters_reshaped)

        except ValueError as e:
            if "None as outputs_info" in e.message:
                # TODO Make this error more interpretable
                raise ValueError("Mismatch between number of dynamic outputs and number of dynamic inputs. \n"
                                 "Make sure function outputs and inputs match (i.e. all dynamic inputs have a corresponding\n"
                                 " returned value, and make sure dynamic parameters are correctly set to be dynamic and\n"
                                 " static parameters are set to be static")
            else:
                raise e

        except TypeError as e:
            # Translate PyMC3 / theano error messages
            if 'takes exactly' in e.message:
                # Catch incorrect number of arguments errors
                n_inputs_provided = len(self.model_inputs)
                n_dynamic_provided = len(self.dynamic_parameters_reshaped)
                n_static_provided = len(self.static_parameters_reshaped)
                raise TypeError("Incorrect number of arguments provided to the learning model function. \nFunction takes {0} "
                                "arguments, provided {1} (outcome and time plus {2} additional inputs; \n{3} dynamic parameters;"
                                " {4} static parameters)".format(len(inspect.getargspec(self.learning_model)[0]),
                                                                 2 + n_inputs_provided + n_dynamic_provided + n_static_provided,
                                                                 n_inputs_provided,
                                                                 n_dynamic_provided + 1, n_static_provided))
            elif 'Wrong number of inputs for LE.make_node' in e.message:
                got = re.search('(?<=got )\d+', e.message).group()
                expected = re.search('(?<=expected )\d+', e.message).group()
                raise TypeError("A theano function has been given the wrong number of arguments (you provided {0} and "
                                "it expected {1}. Check all theano functions used in the model (e.g. switch, comparisons)"
                                " have the correct number of inputs".format(got, expected))
            elif 'instance' in e.message:
                raise TypeError('{0}\n'
                                'This probably means a variable in the model function is not defined. '
                                'Check for typos in argument and variable names')
            else:
                raise e

        if not len(value):  # TODO doesn't work if function only returns tuple of one value
            value = [value]
            _value = [_value]

        value = value[:self.__n_dynamic]  # hack, for some reason non-reused outputs don't join properly

        for n, v in enumerate(value):
            value[n] = T.concatenate([_value[n], v])

        # end awful hack
        observation_dynamics = [value[i] for i in self.__observation_dynamic_inputs]

        if self.observation_model is not None:
            prob = self.observation_model(*observation_dynamics + self.observation_parameters_reshaped)
        else:
            prob = value
        # TODO return everything
        return prob


    def logp(self, x):
        """
        Calls the get_value function and then calculates log likelihood for the model based on estimated probabilities

        Args:
            x: Outcome data

        Returns:
            Log likelihood
        """

        model_output = self.get_value(x)

        for arg, val in self.logp_args.iteritems():
            if isinstance(val, str):

                if val not in self.__learning_return_names and val not in self.__observation_return_names:
                    raise AttributeError("Learning and observation functions "
                                         "has no return named {0}, "
                                         "valid learning return names = {1}, "
                                         "valid observation return names = {2}".format(val, self.__learning_return_names,
                                                                                       self.__observation_return_names))
                else:
                    # TODO documentation for learning vs observation model outputs
                    if self.observation_model is not None:
                        for n, i in enumerate(self.__observation_return_names):
                            if i == val:
                                self.logp_args[arg] = model_output[n]

                    else:
                        for n, i in enumerate(self.__learning_return_names):
                            if i == val:
                                self.logp_args[arg] = model_output[n]


        if self.logp_distribution is None:  # for some reason this gets compiled multiple times
            model_vars = copy.copy(self.vars)
            self.logp_distribution = self.logp_function(**self.logp_args)
            self.logp_vars = [i.name for i in self.vars if i not in model_vars]

        responses_nonan = T.switch(T.isnan(self.responses), 0., self.responses)
        logp = T.sum(self.logp_distribution.logp(self.responses))

        return logp


class DMModel():

    """
    Class used for defining DMpy models

    Args:
        learning_model: Function defining the learning model to be used
        learning_parameters: A list of parameters defined using the Parameter class, given in the order expected by the learning model function
        observation_model: Function defining the observation model to be used. If no observation model is used this can be indicated by providing None
        observation_parameters: A list of parameters defined using the Parameter class, given in the order expected by the observation model function. If no observation model is used this can be indicated by providing None
        name: Optional argument used for labelling the model instance

    """

    def __init__(self, learning_model, learning_parameters, observation_model, observation_parameters,
                 logp_function='beta', logp_args=None, name=''):
        self.name = name
        self.learning_model = learning_model
        self.learning_parameters = learning_parameters
        self.observation_model = observation_model
        self.observation_parameters = observation_parameters
        self.logp_function = logp_function
        self.__observation_dynamic_inputs = []
        self.trace = None
        self.simulated = []
        self._simulate_function = None
        self._model = None
        self.fit_complete = False
        self._fit_method = None
        self._recovery_run = False
        self._hierarchical = False
        self.theano_model_inputs = []
        self.logp_args = logp_args
        self.WAIC = None
        self.responses = None
        if self.logp_args is None:
            self.logp_args = []

        self.__n_learning_returns, self.__learning_returns = n_returns(self.learning_model)

        if self.observation_model is not None:
            self.__n_observation_returns, self.__observation_returns = n_returns(self.observation_model)
        else:
            self.__n_observation_returns, self.__observation_returns = 0, []

        self.__n_dynamic= 0

        for p in self.learning_parameters:
            try:
                if p.dynamic:
                    self.__n_dynamic+= 1
            except AttributeError:
                raise ValueError("One or more parameters are not instances of the RLpackage parameter class.\n"
                                 "Failed with parameter value {0}".format(p))

        if not isinstance(self.logp_function, str) and not callable(self.logp_function):
            raise TypeError("Logp distribution should be either a PyMC3 distribution or one of 'normal', 'bernoulli',"
                            " or 'beta")

        if isinstance(self.logp_function, str) and self.logp_function not in ['normal', 'bernoulli', 'beta']:
            raise ValueError("Logp distribution should be either a PyMC3 distribution or one of 'normal', 'bernoulli',"
                            " or 'beta")


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

        if self.observation_model is not None:
            n_obs_params = len(self.observation_parameters)
            self.__n_obs_dynamic = n_obs_dynamic(self.observation_model, n_obs_params)
        else:
            n_obs_params = 0
            self.__n_obs_dynamic = 0


        # create model
        self._pymc3_model = None


    def _create_model(self, mle=False, hierarchical=False):

        """
        Internally used method for generating PyMC3 model instance - allows model instance to be cached and reused

        Args:
            outcomes: Outcomes (theano shared variable)
            responses: Responses (theano shared variable)
            n_subjects: Number of subjects (theano shared variable)
            n_runs: Number of runs per subject (theano shared variable)
            mle: Specifies whether to use MLE estimation when fitting - if true, parameters are converted to uniform/flat

        """

        # record the initial state of the model, used for filling in starting values
        self.__learning_model_initial = function_wrapper(self.learning_model, self.__n_learning_returns, self.__n_dynamic,
                                                         len(self.model_inputs))

        with pm.Model(theano_config={'compute_test_value': 'ignore', 'mode': 'FAST_RUN', 'exception_verbosity': 'high'}) as model:

            m = _PyMCModel('model', learning_models=(self.learning_model, self.__learning_model_initial),
                          learning_parameters=self.learning_parameters,
                          observation_model=self.observation_model, vars=model.vars,
                          observation_parameters=[self.observation_parameters, self.__observation_dynamic_inputs],
                          responses=self.responses, observed=self.outcomes, outcomes=self.outcomes, time=self.time,
                          n_subjects=self.n_subjects, n_runs=self.n_runs, hierarchical=hierarchical, mle=mle,
                          logp_function=self.logp_function, logp_args=self.logp_args, model_inputs=self.theano_model_inputs)


            self._DMpy_model = m

            params = (m.distribution.learning_parameters, m.distribution.observation_parameters)
            params = [i for j in params for i in j]
            self.params = [p for p in params if p is not None]

            # print m.distribution.unobserved_RVs

        self._pymc3_model = model

        print "Created model"


    def fit(self, responses, outcomes=None, fit_method='MLE', hierarchical=False, plot=True, fit_stats=False, recovery=False,
            exclude_subjects=None, exclude_runs=None, fit_kwargs=None, sample_kwargs=None, suppress_table=False,
            model_inputs=None, response_transform=None):

        """
        General fitting method.

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
            exclude_subjects: List of subject IDs to exclude from model fitting
            exclude_runs: List of runs to exclude from model fitting
            fit_kwargs: Dictionary of keyword arguments passed to underlying MLE, MAP and variational fitting functions. See
            PyMC3 documentation for more details (http://docs.pymc.io/notebooks/getting_started.html)
            sample_kwargs: Dictionary of keyword arguments passed to underlying variational and MCMC sampling functions.
            suppress_table: If set to true, parameter table will not be printed when model fitting is complete.
            model_inputs: Additional inputs to the model.
            response_transform: A function to transform the responses.

        """

        allowed_methods = ['MLE', 'MAP', 'Variational', 'MCMC', 'mle', 'map', 'variational', 'mcmc']

        if recovery:
            self._recovery_run = False

        # Load data
        subjects, n_runs, responses, self.sims, \
        loaded_outcomes, self.model_inputs = load_data(responses, exclude_subjects=exclude_subjects,
                                                       exclude_runs=exclude_runs,
                                                       additional_inputs=model_inputs)

        if outcomes is None and loaded_outcomes is None:
            raise ValueError("No outcomes provided. Please provide outcomes either as an array or as a column in "
                             "the response file named 'Outcome'")

        responses = responses.astype(float)
        if response_transform is not None:
            if not callable(response_transform):
                raise TypeError("Transformation for response variable should be provided as a function, provided type"
                                " {0}".format(type(response_transform)))
            else:
                self.response_transform = response_transform
        elif self.logp_function == 'beta':
            self.response_transform = beta_response_transform  # transform to avoid zeros and ones
        else:
            self.response_transform = lambda x: x

        if loaded_outcomes is not None:
            outcomes = loaded_outcomes
        else:
            outcomes = load_outcomes(outcomes)

        self.subjects = subjects
        n_subjects = len(subjects)

        if fit_kwargs is None:
            fit_kwargs = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        # make sure outcomes and responses look nice

        if responses.shape[0] == 1:
            responses = responses[0]

        if len(responses.shape) < 2:
            responses = responses.reshape(1, responses.shape[0])

        if responses.shape[1] != outcomes.shape[0]:
            raise ValueError("Responses ({0}) and outcomes ({1}) have unequal lengths".format(responses.shape[1],
                                                                                              outcomes.shape[0]))
        if fit_method in ['MLE', 'mle']:
            mle = True
        else:
            mle = False

        time = np.tile(np.arange(0, outcomes.shape[0]), (outcomes.shape[1], 1)).T

        if self._pymc3_model is None or self._fit_method != fit_method.lower() or n_subjects \
                != self.n_subjects or self._hierarchical != hierarchical or \
                len(self.model_inputs) != len(self.theano_model_inputs):

            # create model if it doesn't exist, the fitting method has been changed, or number of subjects has changed
            # turn outcomes and responses into shared variables

            self.logp_function = self.logp_function
            self.responses = theano.shared(responses)
            self.outcomes = theano.shared(outcomes)
            self.theano_model_inputs = [theano.shared(i.astype(np.float64)) for i in self.model_inputs]
            self.time = theano.shared(time)
            self.n_subjects = n_subjects
            self.n_runs = theano.shared(n_runs)
            self._create_model(mle=mle, hierarchical=hierarchical)

        self.responses.set_value(self.response_transform(responses))
        self.outcomes.set_value(outcomes)
        for n, i in enumerate(self.theano_model_inputs):
            i['input'].set_value(self.model_inputs[n])
        self.time.set_value(time)
        self.n_subjects = n_subjects
        self.n_runs.set_value(n_runs)
        self._fit_method = fit_method.lower()
        self._hierarchical = hierarchical

        # run fitting method

        if fit_method in ['MLE', 'mle']:
            # MAP method is used for MLE - only difference is the absence of priors when setting up parameters
            self._fit_MAP(plot=plot, recovery=recovery, suppress_table=suppress_table, mle=True,
                          **fit_kwargs)

        elif fit_method in ['MAP', 'map']:
            self._fit_MAP(plot=plot, recovery=recovery, suppress_table=suppress_table, **fit_kwargs)

        elif fit_method in ['variational', 'Variational']:
            self._fit_variational(plot=plot, hierarchical=hierarchical, recovery=recovery, suppress_table=suppress_table,
                                  fit_stats=fit_stats, fit_kwargs=fit_kwargs, sample_kwargs=sample_kwargs)

        elif fit_method in ['MCMC', 'mcmc']:
            self._fit_MCMC(plot=plot, hierarchical=hierarchical, recovery=recovery, suppress_table=suppress_table,
                           fit_stats=fit_stats, **sample_kwargs)

        else:
            raise ValueError("Invalid fitting method provided ({0}). Fit method should be one of {1}"
                             .format(fit_method, allowed_methods))


    def _fit_MCMC(self, hierarchical=False, plot=True, fit_stats=False, recovery=True, suppress_table=False,
                  **kwargs):

        sns.set_palette("deep")

        print "Fitting model using NUTS"
        start = timer()

        # check data is correct
        # assert len(outcomes) == len(observed), "Outcome and observed data are " \
        #                                        "different lengths ({0} and ({1}".format(len(outcomes), len(observed))

        if hierarchical and self.n_subjects < 2:
            warnings.warn("\nWarning: Hierarchical model fitting only possible with more than one subject, "
                          "fitting individual subject\n")
            hierarchical = False

        elif hierarchical and self.n_subjects > 1:
            print "Performing hierarchical model fitting for {0} subjects".format(self.n_subjects)

        elif not hierarchical and self.n_subjects > 1:
            print "Performing non-hierarchical model fitting for {0} subjects".format(self.n_subjects)

        with self._pymc3_model:

            self.trace = pm.sample(**kwargs)

            if plot:
                traceplot(self.trace)

        self.fit_values = pm.summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()

        self.parameter_table = parameter_table(pm.summary(self.trace), self.subjects, self._DMpy_model.distribution.logp_vars)

        if not suppress_table:
            print "\nPARAMETER ESTIMATES\n"
            print self.parameter_table

        if recovery and self.sims is not None:
            self.recovery_correlations = self.recovery()
        elif recovery:
            warnings.warn("No simulations have been performed, unable to perform parameter recovery tests")

        if fit_stats:
            # these seem to take a lot of time...
            self.fit_stats()


        #self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)
        self.fit_complete = True
        end = timer()
        print "Finished model fitting in {0} seconds".format(end - start)


    def _fit_variational(self, plot=True, hierarchical=True, fit_stats=False, recovery=True, suppress_table=False,
                         fit_kwargs=None, sample_kwargs=None):

        sns.set_palette("deep")

        print "\n-------------------" \
              "Fitting model using ADVI" \
              "-------------------\n"
        start = timer()
        # check data is correct TODO change for multi-subject data
        # assert len(outcomes) == len(observed), "Outcome and observed data are " \
        #                                        "different lengths ({0} and ({1}".format(len(outcomes), len(observed))

        if hierarchical and self.n_subjects < 2:
            warnings.warn("\nWarning: Hierarchical model fitting only possible with more than one subject, "
                          "fitting individual subject\n")
            hierarchical = False

        elif hierarchical and self.n_subjects > 1:
            print "Performing hierarchical model fitting for {0} subjects".format(self.n_subjects)

        elif not hierarchical and self.n_subjects > 1:
            print "Performing non-hierarchical model fitting for {0} subjects".format(self.n_subjects)

        with self._pymc3_model:

            self.approx = fit(**fit_kwargs)
            self.trace = sample_approx(self.approx, **sample_kwargs)

            print "Done"

        if plot:
            traceplot(self.trace)

        self.fit_values = pm.summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()

        self.parameter_table = parameter_table(pm.summary(self.trace), self.subjects, self._DMpy_model.distribution.logp_vars)

        if not suppress_table:
            print "\nPARAMETER ESTIMATES\n"
            print self.parameter_table

        if recovery and self.sims is not None:
            self.recovery_correlations = self.recovery()

        if fit_stats:
            # these seem to take a lot of time...
            self.fit_stats()

        self.fit_complete = True

        # self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)
        end = timer()
        print "Finished model fitting in {0} seconds".format(end - start)


    def _fit_MAP(self, plot=True, mle=False, recovery=True, suppress_table=False, **kwargs):


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

        print "Performing model fitting for {0} subjects".format(self.n_subjects)

        self._model = {}
        self.map_estimate = {}

        with self._pymc3_model:

            try:
                self.map_estimate = find_MAP(**kwargs)
            except ValueError as err:
                warnings.warn("Fitting failed, this is probably because your model returned NaN values")
                raise err

            self.raw_fit_values = self.map_estimate


        self.fit_complete = True
        # need to backwards transform these values

        untransformed_params = {}

        for p in self.params:
            for m in self.raw_fit_values.keys():
                n = re.search('.+(?=_.+__)', m)
                if (n and n.group() == p.name) or m == p.name:
                    if '__' in m:
                        untransformed_params[p.name] = p.backward(self.raw_fit_values[m]).eval()
                    else:
                        untransformed_params[p.name] = self.raw_fit_values[m]

        self.fit_values = untransformed_params

        self.parameter_table = pd.DataFrame(self.fit_values)
        self.parameter_table['Subject'] = self.subjects
        self.parameter_table.sort_values('Subject')

        if recovery and self.sims is not None:
            self.recovery_correlations = self.recovery()

        if not suppress_table:
            print "\nPARAMETER ESTIMATES\n"
            print self.parameter_table

        self.log_likelihood, self.BIC, self.AIC = model_fit(self._pymc3_model.logp_nojac, self.map_estimate,
                                                            self._pymc3_model.vars, self.outcomes, self.n_subjects)

        self.WAIC = None
        end = timer()
        print "Finished model fitting in {0} seconds".format(end-start)


    def fit_stats(self):

        if self.WAIC is None and not self.fit_complete:

            print "Calculating WAIC..."
            self.WAIC = pm.waic(self.trace, self._pymc3_model).WAIC
            print "WAIC = {0}".format(self.WAIC)
            print "Calculated fit statistics"

        elif self.fit_complete:
            print "WAIC = {0}".format(self.WAIC)

        else:
            raise AttributeError("Model has not been fit")


    def individual_fits(self, logp_functions=None, data_type='discrete'):

        """
        Provides model fits for individual subjects. By default this will use the logp function provided for model fitting, but additional functions can be provided for comparison. For example, if fitting using R^2 as the logp function, it may be useful to also check the mean squared error of the fit.

        Args:
            logp_functions: A dictionary of logp functions to use in addition to the one used for model fitting, with the form {'name': (logp_function, additional_arguments)), where 'name' is the name of the logp function (this is simply used to identify the results of the function in the output, logp_function is a predefined logp function, and additional_arguments is a list of additional argument values passed to the logp function. If the function does not take additional arguments, just pass an empty list as the second entry in the tuple.
            data_type: Type of data, either 'discrete' or 'continuous' - used for calculating the BIC correctly. If the logp method during fitting is set to either 'll' or 'r2' this will be set appropriately automatically.

        Returns:
            A dataframe of model fit statistics. The logp function used in model fitting is referred to as "logp"

        """

        if logp_functions == None:
            logp_functions = dict()

        if not isinstance(logp_functions, dict):
            raise TypeError("Logp functions should be provided as a dictionary of the form {'name': (function, additional args)}")

        for k, v in logp_functions.iteritems():
            if len(v) != 2:
                raise ValueError("Wrong number of values provided for logp function {0}, provided {1} values but 2 should"
                                 "be provided: the logp function and a list of additional arguments".format(k, len(v)))

        if self._hierarchical == True:
            raise NotImplementedError("Individual fits not available for hierarchical models")

        elif self.fit_complete == False:
            raise AttributeError("Model has not been fit")

        if data_type not in ['discrete', 'continuous']:
            raise ValueError("Data type should be either 'discrete' or 'continuous', {0} was provided".format(data_type))

        sim_results, _ = self.simulate()

        if 'P' in sim_results['sim_results'].keys():
            predicted = np.vstack(sim_results['sim_results']['P'])
        else:
            predicted = np.vstack(sim_results['sim_results']['value'])

        if len(predicted.shape) == 1:
            predicted.reshape((predicted.shape, 1))

        n_runs = self.n_runs.eval()
        predicted = predicted.reshape(n_runs * predicted.shape[0], predicted.shape[1] / n_runs, order='F')

        true = self.responses.eval().T
        true = true.reshape(n_runs * true.shape[0], true.shape[1] / n_runs, order='F')

        # TODO REDO ALL THIS
        if self.logp_function == 'r2':
            logp_functions['logp'] = (r2, [])
            data_type = 'continuous'
        elif self.logp_function == 'll':
            logp_functions['logp'] = (log_likelihood, [])
            data_type = 'discrete'
        else:
            logp_functions['logp'] = (self.logp_function, self.logp_args)

        if self.logp_function == 'r2' or data_type == 'continuous':
            logp_functions['rss'] = (rss, [])

        logp_results = dict()

        for k, v in logp_functions.iteritems():
            logp_results[k] = []
            for n, (t, p) in enumerate(zip(true.T, predicted.T)):
                result = v[0](t, p, *v[1])
                if getattr(result, 'eval', None):
                    result = result.eval()  # deal with tensor results
                logp_results[k].append(result)
            logp_results[k] = np.array(logp_results[k])

        o = self.outcomes.eval()
        o = true.reshape(n_runs * o.shape[0], o.shape[1] / n_runs)

        if self.logp_function == 'r2' or data_type == 'continous':
            self.BIC_individual = bic_regression(self._pymc3_model.vars, 1, o, -logp_results['rss'], individual=True)
            self.AIC_individual = bic_regression(self._pymc3_model.vars, 1, o, -logp_results['rss'], individual=True)  # TODO

        if self.logp_function == 'll' or data_type == 'discrete':
            self.BIC_individual = bic(self._pymc3_model.vars, 1, o, logp_results['logp'], individual=True)
            self.AIC_individual = aic(self._pymc3_model.vars, 1, logp_results['logp'])


        fit_table = dict(subject=self.subjects, BIC=self.BIC_individual, AIC=self.AIC_individual)
        for k in logp_results.keys():
            fit_table[k] = logp_results[k]

        fit_table = pd.DataFrame(fit_table)

        self.model_fit_individual = fit_table

        # print "Individual model fit statistics"
        # print self.model_fit_individual

        return self.model_fit_individual


    def tracePlot(self):

        """
        Returns:
            A traceplot
        """

        sns.set_palette("deep")
        traceplot(self.trace)


    def simulate(self, outcomes=None, learning_parameters=None, observation_parameters=None, plot=False,
                 output_file='', n_subjects=1, return_choices=False, combinations=False,
                 plot_against_true=False, noise_mean=0, noise_sd=0, model_inputs=None,
                 response_variable='prob'):

        """
        Args:
            outcomes: Task outcomes (i.e. rewards/punishments). These can be provided in two formats: A 1D numpy array (for a single subject/run), or a pandas dataframe (or path to a saved dataframe) containing at least an 'Outcome' column. The dataframe can optionally include a column named "Run" that specifies different runs of the outcomes (these are repeated for each subject) and a column named "Subject" to specify different subjects.
            learning_parameters: Parameter values for the learning model to be used in simulation. Should be given as a dictionary of the format {'parameter name': parameter values}. Parameter values can be either single values or a list/numpy array of values.
            observation_parameters: Parameter values for the observation model, provided in the same format as those for the learning model
            plot: If true, plots the simulated trajectories
            output_file: Path to save simulated response dataframe to
            n_runs: Number of runs to simulate per subject. Each parameter combination will be simulated as many times as is specified here
            n_subjects: Number of subjects to simulate. Each parameter combination will be simulated as many times as is specified here
            return_choices: Adds a column to the simulation dataframe representing binary choices generated from the response variable
            combinations: If true, every combination of provided parameter values will be simulated
            plot_against_true: If the model has been fit, produces a plot for each subject of their true responses against responses simulated using their best fitted parameter estimates
            noise_mean: Sets the mean of the noise distribution (can be useful for adding a bias to subjects' responses), default = 0
            noise_sd: Sets the standard deviation of the noise distribution. Default = 0, increasing this will add noise.
            model_inputs: List of column names to use as additional inputs for the model. This requires the outcomes to be in dataframe format.
            response_variable: The output of the model to be used as the subject's response when producing a dataframe of the simulated results. The given variable will be renamed "Responses" in the dataframe and this will be used if performing parameter recovery on the simulated data. Default="prob".

        Returns:
            Tuple: (simulated results, output path), where simulated results is an instance of the SimulatedResults class

        """

        self._recovery_run = False

        n_runs = 1

        if model_inputs is None:
            model_inputs = []

        # Check that we either have parameters provided or the model has been fit
        if learning_parameters == None and not self.fit_complete:
            raise ValueError("No parameter values provided and model has not been fit. Must explicitly "
                             "provide parameter values for simulation or fit the model first")

        # Turn the outputs into a dictionary with return names as keys and check that the response variable exists
        return_names = self.__learning_returns + self.__observation_returns

        if response_variable not in return_names:
            raise KeyError("The provided response variable ('{0}') is not one of the outputs returned by either the "
                           "learning or observation model. Possible outputs are {1}".format(response_variable, return_names))

        # Using user-defined parameter values & outcomes

        if not self.fit_complete:  # We're using user-specified parameter values

            params_from_fit = False
            self.subjects = None

            # Check that parameters are provided in the correct format
            parameter_check(learning_parameters, sim=True)

            # Change the parameter dictionary to an ordered dict to preserve order
            self.sim_learning_parameters = OrderedDict(learning_parameters)

            # Check and convert format of observation parameters
            if observation_parameters is not None:
                parameter_check(observation_parameters, sim=True)
                self.sim_observation_parameters = OrderedDict(observation_parameters)
            else:
                self.sim_observation_parameters = OrderedDict()

            # Load outcomes
            outcomes, model_inputs, n_runs_loaded, \
            n_subjects_loaded, outcome_df = load_data_for_simulation(outcomes, model_inputs)

            # If the number of runs/subjects we've loaded is 1, use the number specified in simulate() call instead
            if n_runs_loaded > 1:
                n_runs = n_runs_loaded
            if n_subjects_loaded > 1:
                n_subjects = n_subjects_loaded

        # Using parameter estimates & outcomes from model fit
        else:
            # TODO write tests for this
            params_from_fit = True  # use best values from model fitting if parameter values aren't provided

            # Get necessary info from existing class attributes
            n_runs = self.n_runs.eval()
            n_subjects = self.n_subjects
            outcomes = self.outcomes.eval()
            model_inputs = self.model_inputs

            self.sim_learning_parameters = OrderedDict()
            self.sim_observation_parameters = OrderedDict()

            learning_parameter_names = [i.name for i in self.learning_parameters]
            if self.observation_parameters[0] is not None:
                observation_parameter_names = [i.name for i in self.observation_parameters]
            else:
                observation_parameter_names = None

            # Get fitted parameter values from parameter table
            for p in self.parameter_table.columns:
                if p.replace('mean_', '') in learning_parameter_names:
                    self.sim_learning_parameters[p.replace('mean_', '')] = np.repeat(self.parameter_table[p].values, n_runs)

                elif observation_parameter_names is not None and p.replace('mean_', '') in observation_parameter_names:
                    self.sim_observation_parameters[p.replace('mean_', '')] = np.repeat(self.parameter_table[p].values, n_runs)

            for p in learning_parameter_names:
                if p not in self.sim_learning_parameters.keys():
                    mean = [i.mean for i in self.learning_parameters if i.name == p]
                    self.sim_learning_parameters[p] = np.repeat(mean, n_runs * n_subjects)

            if observation_parameter_names is not None:
                for p in observation_parameter_names:
                    if p not in self.sim_observation_parameters.keys():
                        mean = [i.mean for i in self.observation_parameters if i.name == p]
                        self.sim_observation_parameters[p] = np.repeat(mean, n_runs * n_subjects)

        # Create parameter combinations

        # First convert any single values to lists
        for p, v in self.sim_learning_parameters.iteritems():
            if not hasattr(v, '__len__'):
                self.sim_learning_parameters[p] = [v]

        for p, v in self.sim_observation_parameters.iteritems():
            if not hasattr(v, '__len__'):
                self.sim_observation_parameters[p] = [v]

        # combine learning and observation parameters into a single list - necessary for creating combinations/pairs
        self.__parameter_values = self.sim_learning_parameters.values() + self.sim_observation_parameters.values()

        # Get combinations
        p_combinations, n_subjects = self._create_parameter_combinations(combinations, self.__parameter_values, n_runs,
                                                                         n_subjects, params_from_fit)

        # put combinations of parameters back into dictionaries
        for n, p in enumerate(self.sim_learning_parameters.keys() + self.sim_observation_parameters.keys()):
            if p in self.sim_learning_parameters.keys():
                self.sim_learning_parameters[p] = p_combinations[:, n]
            else:
                self.sim_observation_parameters[p] = p_combinations[:, n]

        # each parameter now has a list of values

        # Set up parameters
        # Parameters need to be given to scan as arrays - here we take them out of their dictionaries and make sure
        # they're in the correct format (i.e. float64)

        sim_dynamic = []
        sim_static = []
        sim_observation = []

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

        if self.observation_model is not None:
            for i in self.observation_parameters:
                match = False
                for p, v in self.sim_observation_parameters.iteritems():
                    if i.name == p:
                        sim_observation.append(v)
                        match = True
                if not match:
                    raise ValueError("Parameter {0} has no value provided".format(i.name))


        # Ensure outcomes and additional model inputs are the right format
        if outcomes.shape[1] < p_combinations.shape[0]:
            warnings.warn("Fewer outcome lists than simulated subjects, attempting to use same outcomes for each "
                          "subject (number of outcome lists = {0}, number of subjects = {1}".format(outcomes.shape[0],
                                                                                                    p_combinations.shape[0]), Warning)
            try:
                # Try to repeat the outcomes we have
                outcomes = np.tile(outcomes, (1, p_combinations.shape[0] / outcomes.shape[1]))
                for n in range(len(model_inputs)):
                    model_inputs[n] = np.tile(model_inputs[n], (1, p_combinations.shape[0] / model_inputs[n].shape[1]))
            except:
                raise ValueError("Unable to repeat outcome arrays to match number of subjects, make sure to either "
                                 "provide outcomes for each subject in a dataframe or make sure the number of "
                                 "simulated subjects is divisible by the number of outcomes. Number of outcome arrays"
                                 " = {0}, number of simulated subjects = {1}".format(outcomes.shape[1], p_combinations.shape[0]))

        if not outcomes.shape[1] == p_combinations.shape[0]:
            raise ValueError("Number of outcome lists provided does not match number of subjects")


        # Create theano shared variables and scan function

        # when simulation is run for the first time, a theano function based on a scan loop is created and stored
        # this gets reused for subsequent simulations (until the model is redeclared) - this saves compiling on each
        # simulation and speeds the process up a lot

        # Define the simulation function if it doesn't already exist
        outputs_info = sim_dynamic + [None] * (self.__n_learning_returns - len(sim_dynamic))
        time = np.tile(np.arange(0, outcomes.shape[0]), (outcomes.shape[1], 1)).T

        if self._simulate_function == None:
            self._define_simulate_function(outputs_info, sim_static, model_inputs, sim_observation)

        # Call the function
        sim_data = self._simulate_function(outcomes, time, *(model_inputs + sim_static +
                                                             [i for i in outputs_info if i is not None] +
                                                             sim_observation))


        # Rename duplicate return names
        for name, count in Counter(return_names).items():
            if count > 1:
                for i in range(count):
                    return_names[return_names.index(name)] = '{0}_{1}'.format(name, i)

        # check we have the same number of keys as values
        assert len(return_names) == len(sim_data), 'Unequal number of return names and simulated outputs, {0} returns,' \
                                                   ' {1} simulated outputs'.format(len(return_names), len(sim_data))
        self._simulation_results_dict = OrderedDict(zip(self.__learning_returns + self.__observation_returns, sim_data))


        # Check for nans and flatten
        for r, v in self._simulation_results_dict.items():
            if np.any(np.isnan(v)):
                warnings.warn("NaNs present in {0}".format(r))
            if np.any(np.isinf(v)):
                warnings.warn("Infs present in {0}".format(r))
            else:
                self._simulation_results_dict[r] = flatten_simulated(v)

        # Convert simulation results to a pandas dataframe
        if self.fit_complete:
            true_responses = self.responses.eval()
        else:
            true_responses = None
        self.simulation_results = simulated_dataframe(self._simulation_results_dict, outcomes, true_responses,
                                                      model_inputs, n_runs, n_subjects, self.subjects,
                                                      self.sim_learning_parameters,  self.sim_observation_parameters,
                                                      self.fit_complete)

        # Create choices

        # Add noise to the response variable
        if noise_sd > 0 or noise_mean > 1:
            self.simulation_results[response_variable + '_clean'] = self.simulation_results[response_variable]
            print "Adding gaussian noise to response variable {0} with mean {1} and SD {2}".format(response_variable,
                                                                                                   noise_mean,
                                                                                                   noise_sd)
            self.simulation_results[response_variable] = _add_noise(self.simulation_results[response_variable],
                                                                    noise_mean, noise_sd, lower_bound=np.min(outcomes),
                                                                    upper_bound=np.max(outcomes))

        # Set response variable
        if self.logp_function == 'bernoulli' or return_choices:
            self.simulation_results['Response'] = generate_choices2(self.simulation_results[response_variable])

        else:
            self.simulation_results['Response'] = self.simulation_results[response_variable]

        self._recovery_run = False

        # Plots

        if plot:
            self.simulation.plot()

        if plot_against_true:
            self.simulation.plot_against_true()

        # Save to csv

        print "Saving simulated results to {0}".format(output_file)
        if len(output_file):
            self.simulation_results.to_csv(output_file, index=False)

        self.simulation = SimulationResults(self.simulation_results, learning_parameters, observation_parameters,
                                            response_variable, self.__learning_returns, self.__observation_returns,
                                            outcomes, n_subjects, n_runs, self.fit_complete, self.responses)

        return self.simulation, output_file


    def _create_parameter_combinations(self, combinations, parameter_values, n_runs, n_subjects, params_from_fit):

        if combinations:  # create combinations of parameters
            # Remove any duplicates
            for n, i in enumerate(parameter_values):
                parameter_values[n] = list(set(i))
            p_combinations = np.array(list(product(*parameter_values)))  # get product
            n_combinations = p_combinations.shape[0]
            p_combinations = np.repeat(p_combinations, n_runs, axis=0)
            p_combinations = np.tile(p_combinations, (n_subjects, 1))

        else:  # get pairs of parameters
            p_combinations = []
            if not all(len(i) == len(parameter_values[0]) for i in parameter_values):
                raise ValueError("Each parameter should have the same number of values")
            else:
                for i in range(len(parameter_values[0])):
                    p_combinations.append([j[i] for j in parameter_values])
            p_combinations = np.array(p_combinations)
            n_combinations = p_combinations.shape[0]
            if not params_from_fit:
                p_combinations = np.repeat(p_combinations, n_runs, axis=0)
                p_combinations = np.tile(p_combinations, (n_subjects, 1))

        # New n_subjects = number of subjects * number of parameter combinations
        if not params_from_fit:
            n_subjects = n_combinations * n_subjects

        print "Simulating data from {0} sets of parameter values".format(len(p_combinations))

        return p_combinations, n_subjects


    def _define_simulate_function(self, outputs_info, sim_static, model_inputs, sim_observation):

        """
        Defines a theano function that is used to simulate data from the model

        """

        sim_static = [np.array([i]) if not isinstance(i, np.ndarray) else i for i in sim_static]
        outputs_info = [np.array([i]) if not isinstance(i, np.ndarray) and i is not None else i for i in outputs_info]

        # define theano tensors
        time_theano = T.matrix("time", dtype='float64')
        outcomes_theano = T.matrix("outcomes", dtype='float64')
        model_inputs_theano = [T.matrix("model_input_{0}".format(n), dtype='float64') for n in range(len(model_inputs))]
        sim_static_theano = []
        outputs_info_theano = []
        sim_observation_theano = []

        for n, i in enumerate(outputs_info):
            if i is None:
                outputs_info_theano.append(None)
            else:
                outputs_info_theano.append(T.vector("outputs_info_{0}".format(n), dtype='float64'))

        for n, i in enumerate(sim_static):
            sim_static_theano.append(T.vector("sim_static_{0}".format(n), dtype='float64'))
        for n, i in enumerate(sim_observation):
            sim_observation_theano.append(T.vector("sim_observation_{0}".format(n), dtype='float64'))
        # sequences for scan should be in format (n_trials, n_subjects)

        # Run the learning model
        value, updates = scan(fn=self.learning_model,
                              sequences=[dict(input=outcomes_theano), dict(input=time_theano)] +
                                        [dict(input=i) for i in model_inputs_theano],
                              outputs_info=outputs_info_theano,
                              non_sequences=sim_static_theano)

        # Not sure what this does but it's probably important
        for n, i in enumerate(outputs_info_theano):
            if i is not None:
                value[n] = T.vertical_stack(i.reshape((1, i.shape[0])), value[n][:-1, :])

        # Run the observation model
        # TODO change this so it returns a list rather than P + [obs outs] - change documentation
        if self.observation_model is not None:
            observation_dynamics = [value[i] for i in self.__observation_dynamic_inputs]
            obs_outs = self.observation_model(*observation_dynamics + sim_observation_theano)

        else:
            obs_outs = []

        # Combine learning and observation model outputs into a single list
        out = T.as_tensor_variable(list(value) + list(obs_outs))
        self._simulate_function = theano.function(inputs=[outcomes_theano, time_theano] + model_inputs_theano +
                                                         sim_static_theano +
                                                         [i for i in outputs_info_theano if i is not None] +
                                                         sim_observation_theano,
                                                  outputs=out, updates=updates)


    def recovery(self, correlations=True, by=None):

        # TODO create recovery class to store outputs?

        if self.sims is None:
            raise AttributeError("Response file provided for model fitting does not include simulated parameter values")

        if self.parameter_table is None:
            raise AttributeError("The model has not been fit, this is necessary to run parameter recovery tests")

        sns.set_palette("deep")
        self.sims = self.sims.reset_index(drop=True)
        fit_params = [i for i in self.parameter_table.columns if not 'sd_' in i and not 'mc_error' in i and not 'hpd' in i
                      and not 'Subject' in i and not '_sim' in i]
        if not self._recovery_run:
            # Groupby used because runs are repeated
            self.parameter_table = pd.merge(self.parameter_table, self.sims.groupby('Subject').mean().reset_index(),
                                            on='Subject')
            self._recovery_run = True
        print "Performing parameter recovery tests..."
        parameter_values = []
        parameter_values_sim = []
        n_p_free = len(fit_params)

        fontweight = 'normal'

        if by is not None and by not in self.parameter_table.columns:
            raise ValueError("Parameter given as colour is not in parameter table")

        if by is not None:
            regplot_scatter = False
        else:
            regplot_scatter = True

        # SCATTER PLOTS - CORRELATIONS
        f, axarr = plt.subplots(1, n_p_free, figsize=(2.5 * n_p_free, 3))
        for n, p in enumerate(fit_params):  # this code could all be made far more efficient
            if p.replace('mean_', '') + '_sim' not in self.sims.columns:
                raise ValueError("Simulated values for parameter {0} not found in response file".format(p))
            parameter_values.append(self.parameter_table[p])
            a = p.replace('mean_', '') + '_sim'
            parameter_values_sim.append(self.parameter_table[p.replace('mean_', '') + '_sim'])
            if n_p_free > 1:
                ax = axarr[n]
            else:
                ax = axarr

            alpha = np.min([1. / len(self.parameter_table) * 10, 1])

            if self._fit_method in ['variational', 'Variational', 'mcmc', 'MCMC']:
                # marker size proportional to SD if using variational or MCMC
                sns.regplot(self.parameter_table[p.replace('mean_', '') + '_sim'], self.parameter_table[p], ax=ax,
                            scatter_kws={'s': self.parameter_table[p.replace('mean_', 'sd_')] * 500, 'alpha': alpha},
                            scatter=regplot_scatter)
                if by:
                    points = ax.scatter(self.parameter_table[p.replace('mean_', '') + '_sim'], self.parameter_table[p], alpha=0.3,
                                        c=self.parameter_table[by + '_sim'], cmap='plasma',
                                        s=self.parameter_table[p.replace('mean_', 'sd_')] * 500)
                    f.colorbar(points, ax=ax)
            else:

                sns.regplot(self.parameter_table[p.replace('mean_', '') + '_sim'], self.parameter_table[p], ax=ax,
                            scatter=regplot_scatter, scatter_kws={'alpha': alpha})
                if by:
                    points = ax.scatter(self.parameter_table[p.replace('mean_', '') + '_sim'], self.parameter_table[p], alpha=0.3,
                                        c=self.parameter_table[by + '_sim'], cmap='plasma',
                                        lw=1, edgecolors='black')
                    f.colorbar(points, ax=ax)

            eq_line_range = np.arange(np.min([ax.get_ylim()[0], ax.get_xlim()[0]]),
                                      np.min([ax.get_ylim()[1], ax.get_xlim()[1]]), 0.01)
            ax.plot(eq_line_range, eq_line_range, linestyle='--', color='black')
            ax.set_xlabel('Simulated {0}'.format(p), fontweight=fontweight)
            ax.set_ylabel('Estimated {0}'.format(p), fontweight=fontweight)
            ax.set_title('{0}\n'
                         'R2 = {1}'.format(p, np.round(r2_score(self.parameter_table[p.replace('mean_', '') + '_sim'],
                                                          self.parameter_table[p]), 2)), fontweight=fontweight)

            sim_min = np.min(self.parameter_table[p.replace('mean_', '') + '_sim'])
            sim_max = np.max(self.parameter_table[p.replace('mean_', '') + '_sim'])
            true_min = np.min(self.parameter_table[p])
            true_max = np.max(self.parameter_table[p])
            ax.set_xlim([sim_min - np.abs(sim_min) / 10., sim_max + np.abs(sim_max) / 10.])
            ax.set_ylim([true_min - np.abs(true_min) / 10., true_max + np.abs(true_max) / 10.])

            sns.despine()
        if by is None:
            f.suptitle("Simulated-estimated parameter correlations")
        else:
            f.suptitle("Simulated-estimated parameter correlations, coloured by {0}".format(by))
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)

        ## SIMULATED-POSTERIOR CORRELATIONS
        if len(self.parameter_table) > 1 and correlations:
            if np.sum(np.diff(parameter_values_sim)) == 0:
                warnings.warn("Parameter values used across simulations are identical, unable to calculate "
                              "correlations between simulated and estimated parameter values. Try providing a "
                              "range of parameter values when simulating.")
                se_cor = None
            else:
                se_cor = np.corrcoef(parameter_values, parameter_values_sim)[n_p_free:, :n_p_free]
                fig, ax = plt.subplots(figsize=(n_p_free * 1.2, n_p_free * 1))
                cmap = sns.diverging_palette(220, 10, as_cmap=True)
                sns.heatmap(se_cor, cmap=cmap, square=True, linewidths=.5, xticklabels=fit_params,
                            yticklabels=fit_params, annot=True)  # order might not work here
                ax.set_xlabel('Simulated', fontweight=fontweight)
                ax.set_ylabel('Estimated', fontweight=fontweight)
                ax.set_title("Simulated-Posterior correlations", fontweight=fontweight)

                plt.tight_layout()

            ## POSTERIOR CORRELATIONS
            ee_cor = np.corrcoef(parameter_values, parameter_values)[n_p_free:, :n_p_free]
            fig, ax = plt.subplots(figsize=(n_p_free * 1.2, n_p_free * 1))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(ee_cor, cmap=cmap, square=True, linewidths=.5, xticklabels=fit_params,
                        yticklabels=fit_params, annot=True)  # order might not work here
            ax.set_xlabel('Estimated', fontweight=fontweight)
            ax.set_ylabel('Estimated', fontweight=fontweight)
            ax.set_title("Posterior correlations", fontweight=fontweight)
            plt.tight_layout()

        else:
            se_cor = None
            warnings.warn('Only one parameter value provided, cannot perform recovery correlation tests')



        return se_cor  # TODO add ee cor


class Parameter():

    """
    Class used to represent parameters

    """

    def __init__(self, name, distribution, lower_bound=None, upper_bound=None, mean=1., variance=None, dynamic=False,
                 **kwargs):

        """

        Args:
            name: Name of the parameter that will be used in model fitting output
            distribution: String representing distribution  # TODO ADD MORE DISTRIBUTIONS
            lower_bound: Lower bound on the distribution
            upper_bound: Upper bound on the distribution
            mean: Mean of the distribution. If the distribution is specified as 'fixed', this will be used as the value of the parameter
            variance: Variance of the distribution
            dynamic: Specifies whether the parameter is dynamic or static. If the parameter is dynamic, the values provided when declaring the parameter are used as starting values.
            **kwargs: Keyword arguments passed to PyMC3 distributions.
        """

        if distribution == 'uniform' and (lower_bound == None or upper_bound == None):
            raise ValueError("Must specify upper and lower bounds for parameters with uniform distribution")

        elif distribution == 'fixed' and (variance != None and lower_bound != None and upper_bound != None):
            warnings.warn("Parameter is specified as fixed, ignoring variance & bounds")
            self.fixed = True
        elif distribution == 'fixed':
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


class SimulationResults():

    """
    Class used to represent results of simulations

    """

    def __init__(self, results, learning_param_values, observation_param_values, response_variable, learning_returns,
                 observation_returns, outcomes, n_subjects, n_runs, fit_complete, responses):

        self.results = results
        self.learning_param_values = learning_param_values
        self.observation_param_values = observation_param_values
        self.response_variable = response_variable
        self.learning_returns = learning_returns
        self.observation_returns = observation_returns
        self.outcomes = outcomes
        self.n_subjects = n_subjects
        self.n_runs = n_runs
        self.fit_complete = fit_complete
        self.responses = responses

        # Convert tensors to numpy arrays
        if not isinstance(self.outcomes, np.ndarray):
            self.outcomes = self.outcomes.eval()
        if self.responses is not None and not isinstance(self.responses, np.ndarray):
            self.responses = self.responses.eval()


    def plot(self, palette='blues', plot_choices=False, plot_outcomes=True, legend=True, plot_clean=False):

        """
        Plots outputs from simulation. Three figures are created: 1) The response variable, 2) Outputs from the
        learning model, 3) Outputs from the observation model

        Args:
            palette: Seaborn palette used to plot results
            plot_choices: Whether to plot generated binary choices from simulated response variable
            plot_outcomes: Plot task outcomes
            legend: Add a legend to the plots

        """

        # Create three plots - one for response variable, one for learning returns, one for observation returns
        for output in [[self.response_variable], self.learning_returns, self.observation_returns]:

            # Sublots = number of unique parameter combinations X number of runs
            if len(output):

                f, ax = plt.subplots(len(output), self.n_runs, figsize=(6 * self.n_runs, 1.5 * len(output)))

                # Iterate over outputs
                for n, name in enumerate(output):

                    # Iterate over runs
                    for run in range(self.n_runs):

                        pal = sns.color_palette(palette, self.n_subjects)

                        # Plot values
                        ax[n, run].plot(self.results[name][self.results.Run == run], label=name, c=pal[n])

                        if plot_clean and '{0}_clean'.format(name) in self.results.columns:
                            ax[n, run].plot(self.results['{0}_clean'.format(name)][self.results.Run == run],
                                            label=name, c=pal[n])

                        ax[n, run].set_title('{0} - Run {1}'.format(name, run), fontweight='bold')

                        if name == self.response_variable and plot_choices:
                            # Generate and plot choices based on response variable
                            ax[n, run].scatter(np.arange(0, self.results[name][self.results.Run == run]),
                                               generate_choices2(self.results[name][self.results.Run == run]), color='#72a23b',
                                         alpha=0.5, label='Simulated choices')

                        if plot_outcomes:
                            # Plot task outcomes
                            ax[n, run].scatter(np.arange(0, self.results[name][self.results.Run == run]),
                                               self.results['Outcome'][self.results.Run == run], color='#72a23b',
                                         alpha=0.5, label='Outcomes')

                        # Set x and y limits
                        ax[n, run].set_ylim(np.min(self.results[name][self.results.Run == run]) - 0.5,
                                               np.max(self.results[name][self.results.Run == run]) + 0.2)
                        ax[n, run].set_xlim(0, self.results[name][self.results.Run == run].shape[0])

                        ax[n, run].set_xlabel("Trial")

                        if legend:
                            ax[n, run].legend(frameon=True, fancybox=True)

                plt.tight_layout()


    def plot_against_true(self, subjects=None, runs=None, show_outcomes=True):

        """
        Plots results of simulation against actual subjects' behaviour

        Args:
            subjects:

        Returns:

        """

        if isinstance(subjects, str) or isinstance(subjects, int):
            subjects = [subjects]

        if isinstance(runs, int):
            runs = [runs]

        if not self.fit_complete:
            raise AttributeError("Model not fit, unable to plot simulated against true values")

        else:
            if subjects is None:
                n_plot_subjects = self.n_subjects
                subjects = self.results.Subject.unique()
            else:
                n_plot_subjects = len(subjects)

            if runs is None:
                n_plot_runs = int(self.n_runs)
                runs = range(int(self.n_runs))
            else:
                n_plot_runs = len(runs)
            fig, ax = plt.subplots(n_plot_subjects, n_plot_runs, figsize=(6 * n_plot_runs, 1.5 * n_plot_subjects))
            if len(ax.shape) < 2:
                if n_plot_subjects == 1:
                    ax = np.expand_dims(ax, axis=0)
                elif n_plot_runs == 1:
                    ax = np.expand_dims(ax, axis=1)

            for n, sub in enumerate(subjects):

                sub_df = self.results[self.results.Subject == sub].reset_index()

                for nn, run in enumerate(runs):

                    run_df = sub_df[sub_df.Run == run].reset_index()

                    # Plot values
                    ax[n, nn].plot(run_df[self.response_variable], label='Model', color='tab:orange')
                    ax[n, nn].plot(run_df['True_response'], label='Data', color='tab:blue')

                    ax[n, nn].set_title('{0} - Run {1}'.format(sub, run), fontweight='bold')
                    # Plot task outcomes
                    if show_outcomes:
                        ax[n, nn].scatter(np.arange(0, len(run_df[self.response_variable])),
                                           run_df['Outcome'], edgecolors='#353535', facecolors='none',
                                           alpha=0.5, label='Outcomes', linewidth=1)

                    # Set x and y limits
                    ax[n, nn].set_ylim(np.min(run_df['Outcome']) - 0.2,
                                        np.max(run_df['Outcome']) + 0.2)
                    ax[n, nn].set_xlim(0, run_df[self.response_variable].shape[0])
                    ax[n, nn].legend(frameon=True, fancybox=True)
                    ax[n, nn].set_xlabel("Trial")


            plt.tight_layout()

