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
from DMpy.DMpy import generate_pymc_distribution, n_returns, function_wrapper, load_data, load_outcomes, \
    parameter_table, model_fit, generate_choices2, n_obs_dynamic, simulated_responses

sns.set_style("white")
sns.set_palette("Set1")

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


def initialise_parameters(learning_parameters, observation_parameters, n_subjects, mle, hierarchical):

    # give parameters distributions etc

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


class PyMCModel(Continuous):

    def __init__(self, learning_models, learning_parameters, observation_model, observation_parameters, responses, hierarchical,
                 outcomes, mle=False, *args, **kwargs):
        super(PyMCModel, self).__init__(*args, **kwargs)

        self.fit = False

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
            initialise_parameters(self.learning_parameters, self.observation_parameters, self.n_subjects, mle,
                                  hierarchical)

        ## learning models with multiple outputs
        ## check number of dynamic parameters, if number of learning function outputs is longer, add nones to outputs info

        self.__n_dynamic = len(self.dynamic_parameters)
        self.__n_learning_returns, _ = n_returns(self.learning_model)


    def get_value(self, x):

        # need to make sure order of arguments is right
        # OR figure out a way to match scan arguments to function arguments

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


    def logp(self, x):  # x = outcome data
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
        self.fit = False

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
        
        #############
        


    def fit_MCMC(self, outcomes, responses, hierarchical=False, plot=True, fit_stats=False, **kwargs):

        print "Loading data"

        self.subjects, responses = load_data(responses)
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

        # these seem to take a lot of time...
        if fit_stats:
            print "Calculating DIC..."
            self.DIC = pm.dic(self.trace, rl)
            print "Calculating WAIC..."
            self.WAIC = pm.waic(self.trace, rl)[0]
            print "Calculated fit statistics"


        #self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.fit_values, rl.vars)
        self.fit = True
        end = timer()
        print "Finished model fitting in {0} seconds".format(end - start)


    def fit_variational(self, outcomes, responses, plot=True, draws=100, hierarchical=True, fit_stats=False, **kwargs):

        print "Loading data"

        self.subjects, responses = load_data(responses)
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

            approx = fit(model=rl, **kwargs)
            self.trace = sample_approx(approx, draws=draws)

            print "Done"

            if plot:
                traceplot(self.trace)

        self.fit_values = pm.df_summary(self.trace, varnames=self.trace.varnames)['mean'].to_dict()
        self.fit = True
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


    def fit_MAP(self, outcomes, responses, plot=True, mle=False, **kwargs):

        print "Loading data"

        self.subjects, responses = load_data(responses)
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

        self.fit = True
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
            print self.parameter_table


        # self.logp = rl.logp
        # self.vars = rl.vars
        self.log_likelihood, self.BIC, self.AIC = model_fit(rl.logp, self.map_estimate, rl.vars, outcomes)  # TODO MULTI SUBJECT

        self.DIC = None
        self.WAIC = None
        end = timer()
        print "Finished model fitting in {0} seconds".format(end-start)


    def fit_MLE(self, outcomes, responses, plot=True, **kwargs):

        self.fit_MAP(outcomes=outcomes, responses=responses, plot=plot, mle=True, **kwargs)


    def tracePlot(self):

        traceplot(self.trace)


    def simulate(self, outcomes, learning_parameters=None, observation_parameters=None, plot=False, responses=None,
                 sim_choices=False, plot_outcomes=True, response_file='', n_subjects=1):

        # TODO rework for hierarchical estimates - provide both group and individual parameter based simulations?

        sim_dynamic = []
        sim_static = []
        sim_observation = []
        sim_observation_dynamic = []

        outcomes = load_outcomes(outcomes)

        ## TODO multi-subject observed data

        if learning_parameters == None and observation_parameters == None:
            raise ValueError("Mst explicitly provide parameter values for simulation")

        for n, i in enumerate(self.learning_parameters):
            match = False
            for p, v in learning_parameters.iteritems():
                if p == i.name:
                    if i.dynamic:
                        # sim_dynamic.append(dict(initial=np.float64(v), taps=[-1]))
                        sim_dynamic.append(np.float64(v))
                    else:
                        sim_static.append(np.float64(v))
                    match = True
            if not match:
                raise ValueError("Parameter {0} has no value provided".format(i.name))

        for i in self.observation_parameters:
            match = False
            for p, v in observation_parameters.iteritems():
                if i.name == p:
                    sim_observation.append(np.float64(v))
                    match = True
            if not match:
                raise ValueError("Parameter {0} has no value provided".format(i.name))



        ## learning models with multiple outputs
        ## check number of dynamic parameters, if number of learning function outputs is longer, add nones to outputs info

        # simulate with scan
        #
        outcomes = np.array(outcomes)
        temp_outcomes = np.hstack([outcomes, 2])  # add an extra outcome to get outputs for the final trial if needed (e.g. PE)

        print sim_dynamic

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

        self.simulated.append((learning_parameters, observation_parameters, result_dict))

        return_names = [i.replace('[', '') for i in return_names]
        return_names = [i.replace(']', '') for i in return_names]

        if plot:

            fig, axarr = plt.subplots(len(eval_value), 1, figsize=(8, 1.5 * len(eval_value)))

            axarr[0].plot(eval_value[0], color='#add0e4', label='V')
            axarr[0].plot(prob, color='#348ABD', label='P')
            axarr[0].set_title('Choice probability', fontweight='bold')


            if sim_choices:
                axarr[0].scatter(np.arange(0, len(prob)), generate_choices2(prob), color='#72a23b', alpha=0.5,
                                 label='Simulated choices')
            if responses is not None:
                axarr[0].scatter(np.arange(0, len(prob)), responses, color='#f18f01', alpha=0.5, label='Observations')
            if plot_outcomes:
                axarr[0].scatter(np.arange(0, len(prob)), outcomes, color='#a23b72', alpha=0.5,
                                 label='Outcomes')

            axarr[0].set_xlim(0, len(prob))
            axarr[0].set_ylim(np.min(outcomes) - 0.5, np.max(outcomes) + 0.2)
            axarr[0].legend(frameon=False, ncol=3, bbox_to_anchor=(0.8, 0.6), loc='lower center')

            for i in range(1, len(eval_value)):
                axarr[i].plot(eval_value[i], color='#348ABD')
                axarr[i].set_xlim(0, len(prob))
                axarr[i].set_title(return_names[i], fontweight='bold')

            plt.tight_layout()

        choices = []

        for i in range(n_subjects):
            choices.append(generate_choices2(prob))

        print "Finished simulating"

        if len(response_file):
            simulated_responses(choices, response_file, learning_parameters, observation_parameters)
            return response_file
        elif sim_choices:
            return choices[0]
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

        self.transform_method = None

    def transform(self, name, transform):

        new_parameter = Parameter(self.name, self.distribution, self.lower_bound, self.upper_bound, self.mean,
                                  self.variance, self.dynamic)

        new_parameter.name = name

        new_parameter.transform_method = transform

        return new_parameter



