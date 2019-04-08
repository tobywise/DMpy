import numpy as np
import theano.tensor as T
from pymc3 import Model, Normal, HalfNormal, DensityDist, Potential, Bound, Uniform, fit, sample_approx, \
    Flat, Deterministic, Minibatch, tt_rng, floatX, Beta, Lognormal
import warnings
import inspect
import re
import pandas as pd
import sys
import os
from io import BytesIO as StringIO
import contextlib
import urllib2
import copy

def get_transforms(bounded_parameter):

    return bounded_parameter.pymc_distribution.transformation.backward, \
           bounded_parameter.pymc_distribution.transformation.forward


def generate_pymc_distribution(p, n_subjects=None, hierarchical=False, mle=False, offset=False, minibatch=0, total_size=0):

    """
    Turns parameters into pymc3 parameter distributions for model fitting
    """
    if hasattr(p, '_Parameter__pymc_kwargs'):
        kwargs = p._Parameter__pymc_kwargs
    else:
        kwargs = {}

    if mle and (p.distribution != 'uniform' and p.distribution != 'flat' and p.distribution != 'fixed'):

        if p.upper_bound is not None and p.lower_bound is not None:
            print("\nParameter {0} distribution is {1}, converting to uniform with bounds ({2}, {3}) for MLE".format(
                p.name, p.distribution, p.lower_bound, p.upper_bound
            ))
            p.distribution = 'uniform'
        elif p.upper_bound is not None:
            print("\nParameter {0} distribution is {1}, converting to uniform with upper bound {2}) for MLE".format(
                p.name, p.distribution, p.upper_bound
            ))
            p.distribution = 'uniform'
        elif p.lower_bound is not None:
            print("\nParameter {0} distribution is {1}, converting to uniform with lower bound {2}) for MLE".format(
                p.name, p.distribution, p.lower_bound
            ))
            p.distribution = 'uniform'
        elif p.distribution == 'beta':
            print("\nParameter {0} distribution is {1}, converting to uniform with bounds (0.001), 0.999) for MLE".format(
                p.name, p.distribution
            ))
            p.distribution = 'uniform'
        else:
            print("\nParameter {0} distribution is {1}, converting to flat for MLE\n".format(
                p.name, p.distribution
            ))
            p.distribution = 'flat'

    if p.fixed:
        p.pymc_distribution = T.ones((p.size, n_subjects)) * p.mean
        p.pymc_distribution.name = p.name

    else:  # there's probably a cleaner way to do this
        
        #shape information
        if p.size > 1:
            shape = (p.size, n_subjects)
        else:
            shape = n_subjects

        if hierarchical and n_subjects < 2:
            raise ValueError("Hierarchical parameters only possible with > 1 subject")

        # NORMAL DISTRIBUTIONS 

        if p.distribution == 'normal' and p.lower_bound is not None and p.upper_bound is not None:
            BoundedNormal = Bound(Normal, lower=p.lower_bound, upper=p.upper_bound)
            if hierarchical:
                if not offset:
                    p.pymc_distribution = BoundedNormal(p.name,
                                                        mu=BoundedNormal(p.name + '_group_mu', mu=p.mean, sd=p.variance),
                                                        sd=Uniform(p.name + '_group_sd', lower=0, upper=100),  # TODO need to allow adjustment of these values somehow
                                                        shape=n_subjects, **kwargs)
                else:
                    offset_param = Normal(p.name + '_offset', mu=0, sd=1, shape=n_subjects)
                    group_mu = Normal(p.name + '_group_mu', mu=p.mean, sd=p.variance)
                    group_sd = Uniform(p.name + '_group_sd', lower=0, upper=10)
                    Am = group_mu + offset_param * group_sd
                    p.pymc_distribution = Deterministic(p.name, (p.upper_bound - p.lower_bound) * T.nnet.sigmoid(Am) + p.lower_bound, **kwargs)
            elif n_subjects > 1:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects, **kwargs)
            else:  # is this necessary?
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, **kwargs)
            # p.backward, p.forward = get_transforms(p)

        elif p.distribution == 'normal' and p.lower_bound is not None:
            BoundedNormal = Bound(Normal, lower=p.lower_bound, upper=p.upper_bound)
            if hierarchical:
                if not offset:
                    p.pymc_distribution = BoundedNormal(p.name,
                                                        mu=BoundedNormal(p.name + '_group_mu', mu=p.mean, sd=p.variance),
                                                        sd=Uniform(p.name + '_group_sd', lower=0, upper=100),  # TODO need to allow adjustment of these values somehow
                                                        shape=n_subjects, **kwargs)
                else:
                    offset_param = Normal(p.name + '_offset', mu=0, sd=1, shape=n_subjects)
                    group_mu = Normal(p.name + '_group_mu', mu=p.mean, sd=p.variance)
                    group_sd = Uniform(p.name + '_group_sd', lower=0, upper=10)
                    Am = group_mu + offset_param * group_sd
                    p.pymc_distribution = Deterministic(p.name, (99999999999 - p.lower_bound) * T.nnet.sigmoid(Am) + p.lower_bound, **kwargs)
            elif n_subjects > 1:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects, **kwargs)
            else:  # is this necessary?
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, **kwargs)
            # p.backward, p.forward = get_transforms(p)

        elif p.distribution == 'normal':
            if hierarchical:
                if not offset:
                    p.pymc_distribution = Normal(p.name,
                                                        mu=Normal(p.name + '_group_mu', mu=p.mean, sd=p.variance),
                                                        sd=Uniform(p.name + '_group_sd', lower=0, upper=100),  # TODO need to allow adjustment of these values somehow
                                                        shape=n_subjects, **kwargs)
                else:
                    offset_param = Normal(p.name + '_offset', mu=0, sd=1, shape=n_subjects)
                    group_mu = Normal(p.name + '_group_mu', mu=p.mean, sd=p.variance)
                    group_sd = Uniform(p.name + '_group_sd', lower=0, upper=10)
                    p.pymc_distribution = Deterministic(p.name, group_mu + offset_param * group_sd, **kwargs)
            elif n_subjects > 1:
                p.pymc_distribution = Normal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects, **kwargs)
            else:  # is this necessary?
                p.pymc_distribution = Normal(p.name, mu=p.mean, sd=p.variance, **kwargs)
            # p.backward, p.forward = get_transforms(p)

        # BETA
        elif p.distribution == 'beta':
            if any([p.beta is not None and p.alpha is not None]) and any([p.mean is not None and p.variance is not None]):
                raise AttributeError("Both mean/variance and alpha/beta provided - must provide only one of these parameterisations")
            elif p.mean is not None:
                if hierarchical:
                    if not offset:
                        p.pymc_distribution = Beta(p.name,
                                                            mu=Beta(p.name + '_group_mu', mu=p.mean, sd=p.variance),
                                                            sd=HalfNormal(p.name + '_group_sd', sd=0.05),  # TODO need to allow adjustment of these values somehow
                                                            shape=n_subjects, **kwargs)
                    else:
                        offset_param = Normal(p.name + '_offset', mu=0, sd=1, shape=n_subjects)
                        group_mu = Beta(p.name + '_group_mu', mu=p.mean, sd=p.variance)
                        group_sd = Uniform(p.name + '_group_sd', lower=0, upper=10)
                        p.pymc_distribution = Deterministic(p.name, group_mu + offset_param * group_sd, **kwargs)
                elif n_subjects > 1:
                    p.pymc_distribution = Beta(p.name, mu=p.mean, sd=p.variance, shape=n_subjects, **kwargs)
                else:  # is this necessary?
                    p.pymc_distribution = Beta(p.name, mu=p.mean, sd=p.variance, **kwargs)
            else:
                if hierarchical:
                    if not offset:
                        p.pymc_distribution = Beta(p.name, alpha=Lognormal(p.name + '_group_alpha', 1, 5), beta=Lognormal(p.name + '_group_beta', 1, 5),
                                                            shape=n_subjects, **kwargs)
                    else:
                        # TODO
                        offset_param = Normal(p.name + '_offset', mu=0, sd=1, shape=n_subjects)
                        group_mu = Beta(p.name + '_group_mu', mu=p.mean, sd=p.variance)
                        group_sd = Uniform(p.name + '_group_sd', lower=0, upper=10)
                        p.pymc_distribution = Deterministic(p.name, group_mu + offset_param * group_sd, **kwargs)
                elif n_subjects > 1:
                    p.pymc_distribution = Beta(p.name, alpha=p.alpha, beta=p.beta, shape=n_subjects, **kwargs)
                else:  # is this necessary?
                    p.pymc_distribution = Beta(p.name, alpha=p.alpha, beta=p.beta, **kwargs)

        # UNIFORM

        elif p.distribution == 'uniform':
            if hierarchical:
                p.pymc_distribution = Uniform(p.name, lower=p.lower_bound, upper=p.upper_bound,
                                             shape=n_subjects, **kwargs)
            elif T.gt(n_subjects, 1):
                print(p.size, n_subjects)
                p.pymc_distribution = Uniform(p.name, lower=p.lower_bound, upper=p.upper_bound,
                                             shape=n_subjects, **kwargs)
            else:
                p.pymc_distribution = Uniform(p.name, lower=p.lower_bound, upper=p.upper_bound, **kwargs)
            if hasattr(p.pymc_distribution, "transformation"):
                p.backward, p.forward = get_transforms(p)

        elif p.distribution == 'flat':
            if hierarchical:
                p.pymc_distribution = Flat(p.name, shape=n_subjects, transform=p.transform_method, **kwargs)
            elif n_subjects > 1:
                p.pymc_distribution = Flat(p.name, shape=n_subjects, transform=p.transform_method, **kwargs)
            else:
                p.pymc_distribution = Flat(p.name, **kwargs)
            if hasattr(p.pymc_distribution, "transformation"):
                p.backward, p.forward = get_transforms(p)


    # Minibatching
    if minibatch > 0:
        p.pymc_distribution = p.pymc_distribution[tt_rng().uniform(size=(minibatch,), low=0,
                                                                            high=total_size - 1e-10).astype('int64')]

    return p


def generate_choices2(pa):

    """
    Simulates choices based on choice probabilities
    """

    return (np.random.random(pa.shape) < pa).astype(int)


def backward(a, b, x):
    a, b = a, b
    r = (b - a) * np.exp(x) / (1 + np.exp(x)) + a
    return r


def bic(variables, n_subjects, outcomes, likelihood, individual=False):

    if not individual:
        BIC = (len(variables) * n_subjects) * np.log(np.prod(outcomes.shape.eval())) - 2. * likelihood

    else:
        BIC = len(variables) * np.log(outcomes.shape[0]) - 2. * likelihood

    return BIC


def bic_regression(variables, n_subjects, outcomes, likelihood, individual=False):

    if not individual:
        n = np.prod(outcomes.shape.eval())
        BIC = n + n * np.log(2 * np.pi) + n * \
              np.log((-likelihood.astype(np.float)) / n) + np.log(n) * (len(variables) * n_subjects)

    else:
        n = outcomes.shape[0]
        BIC = n + n * np.log(2 * np.pi) + n * \
              np.log(-likelihood.astype(np.float) / n) + np.log(n) * len(variables)

    return BIC


def aic(variables, n_subjects, likelihood):

    AIC = 2. * (len(variables) * n_subjects - likelihood)

    return AIC


def model_fit(logp, fit_values, variables, outcome, n_subjects):

    """
    Calculates model fit statistics (log likelihood, BIC, AIC)
    """
    print "calculating fit stats"

    variable_names = [i.name for i in variables]

    for k in fit_values.keys():
        if k not in variable_names:
            fit_values.pop(k)

    likelihood = logp(fit_values)

    BIC = bic_regression(variables, n_subjects, outcome, likelihood)
    AIC = aic(variables, n_subjects, likelihood)

    print "Model likelihood = {0}, BIC = {1}, AIC = {2}".format(likelihood, BIC, AIC)

    return likelihood, BIC, AIC


def parameter_table(df_summary, subjects, logp_rvs):

    """
    Attempts to turn the pymc3 output into a nice table of parameter values for each subject
    """
    df_summary = df_summary[df_summary.index != 'eeee']
    df_summary = df_summary[['mean', 'sd', 'mc_error', 'hpd_2.5', 'hpd_97.5']]
    df_summary = df_summary.reset_index()
    df_summary = df_summary[~df_summary['index'].str.contains('group')]
    df_summary = df_summary[~df_summary['index'].isin(logp_rvs)]

    n_subjects = len(subjects)
    n_parameters = len(df_summary) / n_subjects
    subject_column = pd.Series(np.tile(subjects, n_parameters))
    df_summary['Subject'] = subject_column.values
    if len(subjects) > 1:
        df_summary['index'] = pd.Series([re.search('.+(?=__)', i).group() for i in df_summary['index']]).values

    df_summary = df_summary.pivot(index='Subject', columns='index')
    df_summary.columns = df_summary.columns.map('_'.join)
    df_summary = df_summary.reset_index()

    return df_summary


def load_outcomes(data):

    if type(data) == str:
        try:
            outcomes = np.loadtxt(data)
        except:
            raise ValueError("Outcomes not provided in the right format")

    elif type(data) == np.ndarray or type(data) == list:
        outcomes = data

    else:
        raise ValueError("Outcome data must be either filename, numpy array, or list")

    return outcomes


def n_returns(f):

    try:
        return_code = inspect.getsourcelines(f)[0][-1].replace('\n', '')
    except:
        raise SyntaxError("Unable to determine model outputs, make sure the return of the function is \n"
                          "properly defined. Proper format should be \n"
                          "return (value, outputs that are reused at the next step, other outputs that get saved)")
    n = len(return_code.split(','))
    try:
        if n > 1:
            returns = re.search('(?<=return ).+', ''.join(inspect.getsourcelines(f)[0]
                                                          ).replace('\n', '')[1:]).group()[1:-1].replace(' ', '').split(',')
        else:
            returns = re.search('(?<=return ).+', return_code).group()
    except:
        warnings.warn("Could not retrieve function return names")
        returns = None

    returns = [i for i in returns if len(i)]

    return n, returns


def n_obs_dynamic(f, n_obs_params):

    return len(inspect.getargspec(f)[0]) - n_obs_params


def function_wrapper(f, n_returns, n_reused=1, n_model_inputs=0):

    """
    Wraps user-defined functions to return unprocessed inputs

    Args:
        f: Function
        n_returns: Number of return values given by the function
        n_reused: Number of return values that are re-entered into the function at the next time step

    Returns:
        wrapped: The wrapped function

        Seems to just return the first argument value (should be the value argument) as many times as there are outputs?

    """

    def wrapped(*args):
        outs = [args[0]] * n_returns
        outs[0:n_reused] = args[n_model_inputs + 2:n_reused + n_model_inputs + 2]
        outs = tuple(outs)

        return outs

    return wrapped


def flatten_simulated(simulated):
    """
    Reshapes a 3-D simulated data array (trials X variable size X subjects) to 2-D.

    Args:
        simulated: An output from simulation

    Returns:
        A 2-D version of the supplied output

    """

    return simulated.transpose(2, 0, 1).reshape((simulated.shape[0] * simulated.shape[2], -1))
    # return simulated.flatten(order='F')

def split_3d_array_columns(simulation_results, k):
    """
    Given a dictionary and key, reduces dimensions to 2_D then splits the value of that key into multiple key value pairs for each column
    E.g. dict['value'].shape == (600, 4) becomes dict['value0'].shape == (600) with 3 other keys of the same shape ('value1', 'value2', 'value3')

    Args:
        simulation_results: Dictionary containing results from simulation
        k: Dictionary key

    """

    # Check for NaNs and Infs
    if np.any(np.isnan(simulation_results[k])):
        warnings.warn("NaNs present in {0}".format(k))
    if np.any(np.isinf(simulation_results[k])):
        warnings.warn("Infs present in {0}".format(k))

    # Reduce to 2D
    simulation_results[k] = flatten_simulated(simulation_results[k])

    # Split columns
    if simulation_results[k].shape[1] > 1:
        for i in range(simulation_results[k].shape[1]):
            simulation_results[k + str(i)] = simulation_results[k][:, i]
        simulation_results.pop(k)
    else:  # Remove unnecessary dimension
        simulation_results[k] = simulation_results[k].squeeze()

def simulated_dataframe(simulation_results, outcomes, responses, model_inputs, n_runs, n_subjects, subjects,
                        learning_parameters, observation_parameters, fit_complete):

    """
    Turns output of simulation into response file
    """

    # Add outcomes and model inputs to the array
    simulation_results['Outcome'] = outcomes
    for n, i in enumerate(model_inputs):
        simulation_results['sim_model_input_{0}'.format(n)] = i

    # Give the dictionary different keys for each element of the output array
    for k in simulation_results.keys():
        split_3d_array_columns(simulation_results, k)
        
    # Turn the result dictionary into a dataframe
    simulated_df = pd.DataFrame(simulation_results)

    # Get the number of trials per run
    n_trials = outcomes.shape[0]

    # Generate "subject" ids and run numbers for simulated data output
    if not fit_complete:

        subject_ids = np.arange(0, n_subjects)
        subject_ids = np.repeat(subject_ids, n_runs)
        subject_ids = ['Subject_' + str(i).zfill(len(str(subject_ids.max()))) for i in subject_ids]
        subject_ids = np.repeat(subject_ids, n_trials)

        run_ids = np.arange(0, n_runs)
        run_ids = np.tile(run_ids, n_subjects)
        run_ids = ['Run_' + str(i).zfill(len(str(run_ids.max()))) for i in run_ids]
        run_ids = np.repeat(run_ids, n_trials)

    # If using values from model fit, use subject IDs
    else:
        subject_ids = np.repeat(subjects, n_runs * n_trials)
        run_ids = np.tile(np.repeat(np.arange(0, n_runs), n_trials), n_subjects)

    # Add subject and run IDs to the dataframe
    simulated_df['Subject'] = subject_ids
    simulated_df['Run'] = run_ids

    # Add true responses # TODO multidim
    if fit_complete:
        simulated_df['True_response'] = responses.flatten()

    # add parameter columns
    for p, v in learning_parameters.items():
        v = np.stack(v).squeeze()
        if v.ndim > 1:
            v = v[:, 0]  # if this is a multidimensional array, take the first value (currently no support for different values across second dimension)
        simulated_df[p + '_sim'] =  np.repeat(v, n_trials)
    for p, v in observation_parameters.items():
        v = np.stack(v).squeeze()
        if v.ndim > 1:
            v = v[:, 0]
        simulated_df[p + '_sim'] = np.repeat(v, n_trials)


    return simulated_df


@contextlib.contextmanager
def stdoutIO(stdout=None):
    # code from https://stackoverflow.com/questions/3906232/python-get-the-print-output-in-an-exec-statement
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old
#

def model_check(model_function, parameters):

    if not isinstance(parameters, dict):
        raise ValueError("Please supply parameters as a dictionary of format {'param_name':value}")

    args = inspect.getargspec(model_function)[0]
    _, returns = n_returns(model_function)

    if len(parameters.keys()) != len(args):
        raise ValueError("Number of supplied parameters ({0}) does not match number of required parameters ({1}).\n\n"
                         "Supplied parameters = {2}\n\n"
                         "Required parameters = {3}".format(len(parameters.keys()), len(args),
                                                            ', '.join(parameters.keys()), ', '.join(args)))

    function_code = inspect.getsource(model_function)
    function_code = re.sub('"""[\s\S\d\D\w\W]*?"""', '', function_code)
    function_code = function_code.split('\n')

    lines = []

    for i in function_code:
        if not 'def ' in i and not 'return' in i:
            i = i.replace(' ', '')
            i = i.replace(' ', '')
            if len(i):
                if i[0] != '#':
                    lines.append(i)

    for arg in args:
        try:
            print arg, parameters[arg]
            def_string = '{0}={1}'.format(arg, parameters[arg])
            lines.insert(0, def_string)
        except KeyError:
            raise KeyError("Parameter {0} not found in supplied parameter dictionary.\n\n"
                           "Supplied parameters = {1}".format(arg, ', '.join(parameters.keys())))

    lines.insert(0, 'import numpy as np')
    lines.insert(0, 'import theano.tensor as T')

    for line in lines:
        print "Running code:\n" \
              "{0}".format(line)
        vars = re.search('.*?(?=[\+\-\*\/\=])', line)
        try:
            exec(line, {'os': '', 'shutil': '', 'sys': ''}, globals())  # evaluate code, making sure user can't do anything stupid
        except NameError:
            raise ValueError("This function doesn't cope well with functions/builtins - try providing them as strings,\n"
                             "e.g. 'np.inf' instead of np.inf")
        if vars:
            with stdoutIO() as s:
                exec("print {0}".format(vars.group()))
            out = s.getvalue().replace('\n', '')
            if not re.match('.+[}a-zA-Z]\.0', out):  # output is a tensor, need to eval
                print "{0}\n".format(out)
            else:
                exec('print {0}.eval()\nprint " "'.format(vars.group()))

    print "RETURNS"

    for i in returns:
        print i
        try:
            exec('print {0}.eval()'.format(i))
        except:
            exec ('print {0}'.format(i))


def r2_individual(true, predicted):

    sst = np.power(true - true.mean(axis=0), 2).sum(axis=0)
    ssr = np.power(true - predicted, 2).sum(axis=0)

    r2 = 1 - ssr / sst

    r2[np.isinf(r2)] = 1

    return r2


def rss_individual(true, predicted):

    rss = np.power(true - predicted, 2).sum(axis=0)

    return rss


def log_likelihood_individual(true, predicted):

    return (np.log(predicted[true.nonzero()]).sum(axis=0) +
     np.log(1 - predicted[(1 - true).nonzero()]).sum(axis=0))


def load_example_outcomes():

    file = urllib2.urlopen(
        'https://raw.githubusercontent.com/tobywise/DMpy/master/docs/notebooks/examples/example_outcomes.txt')

    return np.loadtxt(file)


def beta_response_transform(responses):

    responses = (responses * (np.prod(responses.shape) - 1) + 0.5) / np.prod(responses.shape)

    return responses

def beta_response_transform_t(responses):

    responses = (responses * (T.prod(responses.shape) - 1) + 0.5) / T.prod(responses.shape)

    return responses


def _check_column(data, column, missing=True, equal='Outcome'):

    """
    Convenience function for checking columns in dataframes. Checks for missing data and unequal numbers.

    Args:
        data: Dataframe
        column: Column in dataframe to be checked
        missing: Check for missing values; bool
        equal: Check for equal numbers in another column - i.e. check for equal numbers of outcomes per individual in
        a subject column

    """

    if len(equal) and not equal in data.columns:  # If we have multiple outcome columns
        equal = equal + '0'

    if missing:
        if np.any(data[column].isnull()):
            raise ValueError("Nans present in {0} column".format(column))

    if len(equal) > 0:
        if np.any(np.diff([len(data[equal][data[column] == i]) for i in data[column].unique()]) != 0):
            raise ValueError("Individual values in {0} columns have different numbers of values in {1} columns, "
                             "these should all the the same".format(column, equal))


def get_column_data(data, column, n_subjects, n_runs):

    # Single column
    if column in data.columns:
        _check_column(data, column, missing=True, equal='')
        results = data[column].values
        results = results[:, np.newaxis]
    # Multiple columns
    elif sum([column in i for i in data.columns]) > 1:  # We have more than one outcome
        results = get_multiple_columns(data, column)
    else:
        results = None
    results = np.split(results, n_subjects * n_runs)  # Split into list of runs
    # print(results.shape)
    results = np.dstack(results)

    if column in data.columns:
        results = results.reshape((1, ) + results.shape)

    return results

def get_multiple_columns(data, type=''):
    pat = '{0}[0-9]+'.format(type)
    ids = [int(re.search('[0-9]+', i).group()) for i in data.columns if re.match(pat, i)]  # Find columns
    ids = sorted(ids)  # Sort column ids
    if not 0 in ids:  # Ensure they start from zero
        raise ValueError("{0} column IDs should start from zero. Found IDs {1}".format(type, ids))
    if not np.all(np.diff(ids) == 1):  # Ensure they increase by 1
        raise ValueError("{0} columns should be sequentially numbered, i.e. ['{0}1, {0}2, ...]. Column"
                         " IDs provided = {1}".format(type, ids))
    print("Found {0} {1} columns in data".format(len(ids), type))
    _check_column(data, ['{0}{1}'.format(type, id) for id in ids], missing=True, equal='')
    result = data[['{0}{1}'.format(type, id) for id in ids]].values  # Get values

    return result


def load_data_for_simulation(outcomes, model_inputs=()):

    """
    Load outcomes and check they're in the right format with no missing data

    Args:
        outcomes: Outcomes
        model_inputs: Additional model inputs, list of strings

    Returns:
        outcomes: Outcomes as a numpy array of shape (number of trials, number of runs * number of subjects)
        model_inputs: Additional model inputs, list of arrays in same shape as outcomes
        n_runs: Number of runs per subject
        n_subjects: Number of subjects
        outcome_df: Outcome dataframe (if provided as as dataframe)

    """

    n_runs = 1  # Return one run unless more runs are specified
    n_subjects = 1  # Return one run unless more subjects are specified
    value_list = []  # a list to hold outcome and additional input values extracted from outcome dataframe

    if not len(model_inputs): model_inputs = []

    # Check outcome format
    if isinstance(outcomes, list):
        try:
            outcomes = np.array(outcomes)
        except:
            raise TypeError("Unable to convert outcomes to numpy array. Make sure they are in the correct format")

    if isinstance(outcomes, np.ndarray):

        # Check for incorrectly given model inputs
        if len(model_inputs):
            raise ValueError("Additional inputs can only be specified if outcomes are specified as a dataframe")

        # Check for problems with outcomes
        if any([i == 0 for i in outcomes.shape]):
            raise AttributeError("One outcome array dimension is zero")

        if all([i == 1 for i in outcomes.shape]):
            raise AttributeError("Please provide more than one outcome, current outcomes shape = {0}".format(outcomes.shape))

        if outcomes.ndim > 2:
            raise AttributeError("Outcome arrays should not have more than two dimensions")

        if np.any(np.isnan(outcomes)):
            raise ValueError("Nans present in outcomes")

        if np.any(np.isinf(outcomes)):
            raise ValueError("Inf present in outcomes")

        # Make sure outcomes are in shape (n_trials, 1) if 1D
        if outcomes.ndim == 1:
            outcomes = outcomes.reshape((len(outcomes), 1))

        # Second dimension should represent number of runs
        n_runs = outcomes.shape[1]

        value_list = [outcomes]

        outcome_df = None

    elif isinstance(outcomes, str) or isinstance(outcomes, pd.DataFrame):

        # If a string is provided, it should be the path to a file
        if isinstance(outcomes, str):
            outcome_df = pd.read_csv(outcomes)
        else:
            outcome_df = outcomes

        # Check the dataframe contains everything it should

        if 'Outcome' not in outcome_df.columns and not any(['Outcome' in i for i in outcome_df.columns]):
            raise AttributeError("Outcome data does not contain Outcome column, provided columns = {0}".format(outcome_df.columns))

        if len(outcome_df) < 2:
            raise AttributeError("Outcome dataframe has fewer than two rows")

        # Check for nans in whole dataframe
        if np.any(outcome_df.isnull()):
            raise ValueError("Dataframe contains missing data")

        # Check for missing values and unequal numbers of outcomes per run/subject
        if 'Run' in outcome_df.columns:
            _check_column(outcome_df, 'Run')

        if 'Subject' in outcome_df.columns:
            _check_column(outcome_df, 'Subject')

        # Get number of runs and subjects
        if 'Run' in outcome_df.columns:
            n_runs = len(outcome_df.Run.unique())

        if 'Subject' in outcome_df.columns:
            n_subjects = len(outcome_df.Subject.unique())

        # Check additional inputs
        if not isinstance(model_inputs, list):
            raise TypeError("Model inputs should be provided as a list, provided type {0}".format(type(model_inputs)))

        for n, i in enumerate(model_inputs):
            # Loop over column names and replace them with values in ndarray format
            if not isinstance(i, str):
                raise TypeError("Additional inputs should be strings representing column names in the outcome dataframe, "
                                "item {0} in the list is of type {1}".format(n, type(i)))
            if i not in outcome_df.columns and not any([i in j for j in outcome_df.columns]):
                raise AttributeError("Outcome dataframe has no column named {0}".format(i))

        # Get outcome values and additional input values and turn them into a numpy array
        columns = ['Outcome'] + model_inputs

        # Iterate over outcome column and additional input columns
        for c in columns:
            vals = get_column_data(outcome_df, c, n_subjects, n_runs)
            # Check that things are the right way round
            assert vals.shape[2] == n_subjects * n_runs
            value_list.append(vals)

    else:
        raise TypeError("Outcomes not provided in the correct format. Outcomes should be supplied as either a 1D"
                        " numpy array, a path to a file, or a pandas dataframe: provided type {0}".format(type(outcomes)))

    # Outcomes are the first item in the list
    outcomes = value_list[0]

    # If model inputs are given, these are the subsequent items
    if len(model_inputs):
        model_inputs = value_list[1:]

    # Check that model inputs are the same shape as outcomes
    for n, i in enumerate(model_inputs):
        if model_inputs[n].shape != outcomes.shape:
            raise AttributeError("Model input {0} has shape {1}, "
                                 "outcomes have shape {2}. These should be the same".format(n, model_inputs[n].shape,
                                                                                            outcomes.shape))

    return outcomes, model_inputs, n_runs, n_subjects, outcome_df


def load_data(data_file, exclude_subjects=None, exclude_runs=None, additional_inputs=None):

    # TODO rewrite all of this because it's a mess

    if additional_inputs is None:
        additional_inputs = []
    elif not isinstance(additional_inputs, list):
        raise ValueError("Model inputs provided as {0}, these should be provided as a list".
                         format(type(additional_inputs)))
    else:
        additional_inputs = additional_inputs

    if not isinstance(data_file, pd.DataFrame):
        try:
            data = pd.read_csv(data_file)
        except ValueError:
            raise ValueError("Responses are not in the correct format, ensure they are provided as a .csv or .txt file")
    else:
        data = data_file

    if 'Subject' not in data.columns or not any(['Response' in i for i in data.columns]):
        raise ValueError("Data file must contain the following columns: Subject, Response")

    if exclude_subjects is not None:
        if not isinstance(exclude_subjects, list):
            exclude_subjects = [exclude_subjects]
        data = data[~data.Subject.isin(exclude_subjects)]
        print "Excluded subjects: {0}".format(exclude_subjects)

    subjects = np.unique(data.Subject)
    n_subjects = len(subjects)

    if 'Run' in data.columns and np.max(data['Run']) > 0:
        n_runs = len(np.unique(data['Run']))

        if exclude_runs is not None:
            if not isinstance(exclude_runs, list):
                exclude_runs = [exclude_runs]
            data = data[~data.Run.isin(exclude_runs)]
            print "Excluded runs: {0}".format(exclude_runs)

    else:
        n_runs = 1


    if len(data) % n_subjects:
        raise AttributeError("Unequal number of trials across subjects")

    n_trials = len(data) / (n_runs * n_subjects)

    sim_columns = [i for i in data.columns if '_sim' in i or 'Subject' in i]
    if len(sim_columns) > 1:
        sims = data[sim_columns]
        sim_index = np.arange(0, len(data), n_trials)
        sims = sims.iloc[sim_index]
    else:
        sims = None

    try:
        trial_index = np.tile(range(0, n_trials), n_runs * n_subjects)
    except:
        raise ValueError("Each run must have the same number of trials")

    data['Trial_index'] = trial_index

    if 'Run' in data.columns:
        data['Subject'] = data['Subject'].astype(str) + data['Run'].astype(str)

    # Responses and outcomes
    responses = get_column_data(data, 'Response', n_subjects, n_runs)
    outcomes = get_column_data(data, 'Outcome', n_subjects, n_runs)

    # Other inputs for the model
    additional_input_data = []

    for i in additional_inputs:
        if not any([i in j for j in data.columns]):
            raise AttributeError("Response file has no column containing {0}".format(i))
        input_data = get_column_data(data, i, n_subjects, n_runs)
        additional_input_data.append(input_data)

    if not len(additional_inputs) and any('sim_model_input' in i for i in data.columns):
        for i in data.columns:
            if 'sim_model_input' in i:
                input_data = np.split(data[i].values, n_subjects * n_runs)
                input_data = np.array(input_data).T
                additional_input_data.append(input_data)

    print "Loaded data, {0} subjects with {1} trials * {2} runs".format(n_subjects, n_trials, n_runs)

    return subjects, n_runs, responses, sims, outcomes, additional_input_data

def parameter_check(parameters, sim=False):

    """
    Checks that parameters are specified properly

    Args:
        parameter_dict: Dictionary of parameters
        sim: If true, assumes parameters are being used for simulation and checks for a dictionary of names and values. If false, assumes these are used for fitting and checks for a list of DMPy parameter instances.

    """
    from DMpy.model import Parameter

    if sim:
        if not isinstance(parameters, dict):
            raise TypeError("Parameters should be specified as a dictionary, provided {0}".format(type(parameters)))

        for p, v in parameters.items():
            if not isinstance(v, list) and not isinstance(v, np.ndarray) and not isinstance(v, float) and not\
                    isinstance(v, int):
                raise TypeError(
                    "Parameter {0} is not the correct type. Parameter values should be provided as either a "
                    "list, numpy array or single float/integer, provided type was {1}".format(p, type(v)))

    else:
        if not isinstance(parameters, list):
            raise TypeError("Parameters should be specified as a list, provided {0}".format(type(parameters)))

        for p in parameters:
            if not isinstance(p, Parameter):
                raise TypeError(
                    "Parameter {0} is not the correct type. Parameter values should be provided as a DMPy "
                    "Parameter instance, provided type was {1}".format(p, type(p)))








































