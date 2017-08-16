import numpy as np
from pymc3 import Model, Normal, HalfNormal, DensityDist, Potential, Bound, Uniform, fit, sample_approx, \
    Flat, Deterministic
import warnings
import inspect
import re
import pandas as pd
import time
import os
import sys
from io import BytesIO as StringIO
import contextlib

def get_transforms(bounded_parameter):

    return bounded_parameter.pymc_distribution.transformation.backward, \
           bounded_parameter.pymc_distribution.transformation.forward


def generate_pymc_distribution(p, n_subjects=None, hierarchical=False, mle=False):

    """
    Turns parameters into pymc3 parameter distributions for model fitting
    """

    if mle and (p.distribution != 'uniform' and p.distribution != 'flat' and p.distribution != 'fixed'):

        if p.upper_bound is not None and p.lower_bound is not None:
            print("\nParameter {0} distribution is {1}, converting to uniform with bounds ({2}, {3}) for MLE".format(
                p.name, p.distribution, p.lower_bound, p.upper_bound
            ))
            p.distribution = 'uniform'
        else:
            print("\nParameter {0} distribution is {1}, converting to flat for MLE\n".format(
                p.name, p.distribution
            ))
            p.distribution = 'flat'

    if p.fixed:
        p.pymc_distribution = np.float64(np.ones(n_subjects) * p.mean)

    else:  # there must be a cleaner way to do this

        if hierarchical and n_subjects < 2:
            raise ValueError("Hierarchical parameters only possible with > 1 subject")

        if p.distribution == 'normal' and p.lower_bound is not None and p.upper_bound is not None:
            BoundedNormal = Bound(Normal, lower=p.lower_bound, upper=p.upper_bound)
            if hierarchical:
                p.pymc_distribution = BoundedNormal(p.name,
                                                    mu=BoundedNormal(p.name + '_group_mu', mu=p.mean, sd=p.variance),
                                                    sd=Uniform(p.name + '_group_sd', lower=0, upper=100),  # TODO need to allow adjustment of these values somehow
                                                    shape=n_subjects)
            elif n_subjects > 1:
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects)
            else:  # is this necessary?
                p.pymc_distribution = BoundedNormal(p.name, mu=p.mean, sd=p.variance)
            p.backward, p.forward = get_transforms(p)

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
            p.backward, p.forward = get_transforms(p)

        elif p.distribution == 'normal':
            if hierarchical:
                p.pymc_distribution = Normal(p.name,
                                             mu=Normal(p.name  + '_group_mu', mu=p.mean, sd=p.variance),
                                             sd=Uniform(p.name + '_group_sd', lower=0, upper=100),
                                             shape=n_subjects, transform=p.transform_method)
            elif n_subjects > 1:
                p.pymc_distribution = Normal(p.name, mu=p.mean, sd=p.variance, shape=n_subjects,
                                             transform=p.transform_method)
            else:
                p.pymc_distribution = Normal(p.name, mu=p.mean, sd=p.variance, transform=p.transform_method)
            if hasattr(p.pymc_distribution, "transformation"):
                p.backward, p.forward = get_transforms(p)

        elif p.distribution == 'uniform':
            if hierarchical:
                p.pymc_distribution = Uniform(p.name, lower=p.lower_bound, upper=p.upper_bound,
                                             shape=n_subjects)
            elif n_subjects > 1:
                p.pymc_distribution = Uniform(p.name, lower=p.lower_bound, upper=p.upper_bound,
                                             shape=n_subjects)
            else:
                p.pymc_distribution = Uniform(p.name, lower=p.lower_bound, upper=p.upper_bound)
            if hasattr(p.pymc_distribution, "transformation"):
                p.backward, p.forward = get_transforms(p)

        elif p.distribution == 'flat':
            if hierarchical:
                p.pymc_distribution = Flat(p.name, shape=n_subjects, transform=p.transform_method)
            elif n_subjects > 1:
                p.pymc_distribution = Flat(p.name, shape=n_subjects, transform=p.transform_method)
            else:
                p.pymc_distribution = Flat(p.name)
            if hasattr(p.pymc_distribution, "transformation"):
                p.backward, p.forward = get_transforms(p)

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


def model_fit(logp, fit_values, vars, outcome, individual=True):

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
    if len(subjects) > 1:
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

    # seen = set()
    # subjects = [i for i in data.Subject if i not in seen and not seen.add(i)]
    trial_index = np.tile(range(0, n_trials), n_subjects)
    data['Trial_index'] = trial_index

    data = data.pivot(columns='Subject', values='Response', index='Trial_index')
    subjects = list(data.columns)
    data = data.values.T

    print "Loaded data, {0} subjects with {1} trials".format(n_subjects, n_trials)

    return subjects, data


def load_outcomes(data):

    if type(data) == str:
        try:
            outcomes = np.loadtxt(data)
        except ValueError:
            _, outcomes = load_data(data)

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
            returns = re.search('(?<=\().+(?=\))', return_code).group().split(', ')
        else:
            returns = re.search('(?<=return ).+', return_code).group()
    except:
        warnings.warn("Could not retrieve function return names")
        returns = None

    # code for function with weird returns

    # return_code = inspect.getsourcelines(f)[0][-1].replace('\n', '')
    # return_code = re.search('(?<=isnan\(o\), )\(.*?\)', return_code).group()
    # n = len(return_code.split(','))
    # try:
    #     if n > 1:
    #         returns = re.search('(?<=\().+(?=\))', return_code).group().split(', ')
    #     else:
    #         returns = re.search('(?<=return ).+', return_code).group()
    # except:
    #     warnings.warn("Could not retrieve function return names")
    #     returns = None

    return n, returns


def n_obs_dynamic(f, n_obs_params):

    return len(inspect.getargspec(f)[0]) - n_obs_params


def function_wrapper(f, n_returns, n_reused=1):

    def wrapped(*args):
        outs = [args[0]] * n_returns
        outs[0:n_reused] = args[1:n_reused + 1]
        outs = tuple(outs)

        return outs

    return wrapped


def simulated_responses(simulated, out_file, learning_parameters, observation_parameters):

    """
    Turns output of simulation into response file
    """

    if not isinstance(simulated, list):
        simulated = [simulated]

    responses = []
    subjects = []

    for n, i in enumerate(simulated):
        responses += i.tolist()
        subjects += [n+1] * len(i)

    df = pd.DataFrame(dict(Response=responses, Subject=subjects,
                           Learning_p=str(learning_parameters), Obs_p=str(observation_parameters)))

    # datetime = time.strftime("%Y%m%d-%H%M%S")
    # sim_fname = os.path.join(out_dir, 'simulated_responses_{0}.txt'.format(datetime))

    print "Saving simulated responses to {0}".format(out_file)
    df.to_csv(out_file, index=False)

    return out_file


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


